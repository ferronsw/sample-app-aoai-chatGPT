import os
import json
import logging
import requests
import dataclasses

from typing import List

DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get(
    "AZURE_SEARCH_PERMITTED_GROUPS_COLUMN"
)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


async def format_as_ndjson(r):
    try:
        async for event in r:
            yield json.dumps(event, cls=JSONEncoder) + "\n"
    except Exception as error:
        logging.exception(
            "Exception while generating response stream: %s", error)
        yield json.dumps({"error": str(error)})


def parse_multi_columns(columns: str) -> list:
    if "|" in columns:
        return columns.split("|")
    else:
        return columns.split(",")


def with_max_completion_tokens(model_args: dict) -> dict:
    if "max_tokens" not in model_args or "max_completion_tokens" in model_args:
        return model_args.copy()

    compat_args = model_args.copy()
    compat_args["max_completion_tokens"] = compat_args.pop("max_tokens")
    return compat_args


def with_max_tokens(model_args: dict) -> dict:
    if "max_completion_tokens" not in model_args or "max_tokens" in model_args:
        return model_args.copy()

    compat_args = model_args.copy()
    compat_args["max_tokens"] = compat_args.pop("max_completion_tokens")
    return compat_args


def should_retry_with_max_tokens(error, model_args: dict) -> bool:
    if "max_completion_tokens" not in model_args or "max_tokens" in model_args:
        return False

    error_message = str(error).lower()
    return (
        "max_completion_tokens" in error_message
        and (
            "max_tokens" in error_message
            or "not supported" in error_message
            or "unsupported" in error_message
            or "unknown parameter" in error_message
            or "unrecognized request argument" in error_message
        )
    )


def should_retry_with_max_completion_tokens(error, model_args: dict) -> bool:
    if "max_tokens" not in model_args or "max_completion_tokens" in model_args:
        return False

    error_message = str(error).lower()
    return (
        "max_tokens" in error_message
        and (
            "max_completion_tokens" in error_message
            or "not supported" in error_message
            or "unsupported" in error_message
            or "unknown parameter" in error_message
            or "unrecognized request argument" in error_message
        )
    )


def fetchUserGroups(userToken, nextLink=None):
    # Recursively fetch group membership
    if nextLink:
        endpoint = nextLink
    else:
        endpoint = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$select=id"

    headers = {"Authorization": "bearer " + userToken}
    try:
        r = requests.get(endpoint, headers=headers)
        if r.status_code != 200:
            logging.error(
                f"Error fetching user groups: {r.status_code} {r.text}")
            return []

        r = r.json()
        if "@odata.nextLink" in r:
            nextLinkData = fetchUserGroups(userToken, r["@odata.nextLink"])
            r["value"].extend(nextLinkData)

        return r["value"]
    except Exception as e:
        logging.error(f"Exception in fetchUserGroups: {e}")
        return []


def generateFilterString(userToken):
    # Get list of groups user is a member of
    userGroups = fetchUserGroups(userToken)

    # Construct filter string
    if not userGroups:
        logging.debug("No user groups found")

    group_ids = ", ".join([obj["id"] for obj in userGroups])
    return f"{AZURE_SEARCH_PERMITTED_GROUPS_COLUMN}/any(g:search.in(g, '{group_ids}'))"


def _create_response_envelope(
    response_id,
    model,
    created,
    object_name,
    history_metadata,
    apim_request_id,
):
    return {
        "id": response_id,
        "model": model,
        "created": created,
        "object": object_name,
        "choices": [{"messages": []}],
        "history_metadata": history_metadata,
        "apim-request-id": apim_request_id,
    }


def format_message_response(
    role,
    content,
    history_metadata,
    apim_request_id,
    response_id="",
    model="",
    created=0,
    object_name="chat.completion",
):
    response_obj = _create_response_envelope(
        response_id,
        model,
        created,
        object_name,
        history_metadata,
        apim_request_id,
    )
    response_obj["choices"][0]["messages"].append(
        {
            "role": role,
            "content": content,
        }
    )
    return response_obj


def format_tool_response(
    tool_context,
    history_metadata,
    apim_request_id,
    response_id="",
    model="",
    created=0,
    object_name="chat.completion",
):
    return format_message_response(
        "tool",
        json.dumps(tool_context),
        history_metadata,
        apim_request_id,
        response_id=response_id,
        model=model,
        created=created,
        object_name=object_name,
    )


def format_non_streaming_response(
    chatCompletion,
    history_metadata,
    apim_request_id,
    tool_context=None,
):
    response_obj = _create_response_envelope(
        chatCompletion.id,
        chatCompletion.model,
        chatCompletion.created,
        chatCompletion.object,
        history_metadata,
        apim_request_id,
    )

    if len(chatCompletion.choices) > 0:
        message = chatCompletion.choices[0].message
        if message:
            if tool_context:
                response_obj["choices"][0]["messages"].append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_context),
                    }
                )
            elif hasattr(message, "context"):
                response_obj["choices"][0]["messages"].append(
                    {
                        "role": "tool",
                        "content": json.dumps(message.context),
                    }
                )
            response_obj["choices"][0]["messages"].append(
                {
                    "role": "assistant",
                    "content": message.content,
                }
            )
            return response_obj

    return {}


def format_stream_response(chatCompletionChunk, history_metadata, apim_request_id):
    response_obj = _create_response_envelope(
        chatCompletionChunk.id,
        chatCompletionChunk.model,
        chatCompletionChunk.created,
        chatCompletionChunk.object,
        history_metadata,
        apim_request_id,
    )

    if len(chatCompletionChunk.choices) > 0:
        delta = chatCompletionChunk.choices[0].delta
        if delta:
            if hasattr(delta, "context"):
                messageObj = {"role": "tool",
                              "content": json.dumps(delta.context)}
                response_obj["choices"][0]["messages"].append(messageObj)
                return response_obj
            if delta.role == "assistant" and hasattr(delta, "context"):
                messageObj = {
                    "role": "assistant",
                    "context": delta.context,
                }
                response_obj["choices"][0]["messages"].append(messageObj)
                return response_obj
            else:
                if delta.content:
                    messageObj = {
                        "role": "assistant",
                        "content": delta.content,
                    }
                    response_obj["choices"][0]["messages"].append(messageObj)
                    return response_obj

    return {}


def format_pf_non_streaming_response(
    chatCompletion, history_metadata, response_field_name, citations_field_name, message_uuid=None
):
    if chatCompletion is None:
        logging.error(
            "chatCompletion object is None - Increase PROMPTFLOW_RESPONSE_TIMEOUT parameter"
        )
        return {
            "error": "No response received from promptflow endpoint increase PROMPTFLOW_RESPONSE_TIMEOUT parameter or check the promptflow endpoint."
        }
    if "error" in chatCompletion:
        logging.error(
            f"Error in promptflow response api: {chatCompletion['error']}")
        return {"error": chatCompletion["error"]}

    logging.debug(f"chatCompletion: {chatCompletion}")
    try:
        messages = []
        if response_field_name in chatCompletion:
            messages.append({
                "role": "assistant",
                "content": chatCompletion[response_field_name]
            })
        if citations_field_name in chatCompletion:
            citation_content = {
                "citations": chatCompletion[citations_field_name]}
            messages.append({
                "role": "tool",
                "content": json.dumps(citation_content)
            })

        response_obj = {
            "id": chatCompletion["id"],
            "model": "",
            "created": "",
            "object": "",
            "history_metadata": history_metadata,
            "choices": [
                {
                    "messages": messages,
                }
            ]
        }
        return response_obj
    except Exception as e:
        logging.error(f"Exception in format_pf_non_streaming_response: {e}")
        return {}


def convert_to_pf_format(input_json, request_field_name, response_field_name):
    output_json = []
    logging.debug(f"Input json: {input_json}")
    # align the input json to the format expected by promptflow chat flow
    for message in input_json["messages"]:
        if message:
            if message["role"] == "user":
                new_obj = {
                    "inputs": {request_field_name: message["content"]},
                    "outputs": {response_field_name: ""},
                }
                output_json.append(new_obj)
            elif message["role"] == "assistant" and len(output_json) > 0:
                output_json[-1]["outputs"][response_field_name] = message["content"]
    logging.debug(f"PF formatted response: {output_json}")
    return output_json


def comma_separated_string_to_list(s: str) -> List[str]:
    '''
    Split comma-separated values into a list.
    '''
    return s.strip().replace(' ', '').split(',')
