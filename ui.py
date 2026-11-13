"""Streamlit document workspace with upload, topic history, and streamed chat."""

import json
import os
from collections.abc import Iterator
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


def stream_chat(api_url: str, payload: dict[str, object]) -> Iterator[tuple[str, dict[str, object]]]:
    """Send one request to the SSE endpoint and yield named events in order."""
    request = Request(
        f"{api_url.rstrip('/')}/chat/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urlopen(request, timeout=600) as response:  # noqa: S310 -- local configurable API endpoint.
        event = "message"
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                yield event, json.loads(line.removeprefix("data: "))


def main() -> None:
    """Render the DocuSense document chat workspace."""
    st.set_page_config(
        page_title="DocuSense",
        page_icon=":material/forum:",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
    if not st.session_state.documents_loaded:
        refresh_documents(api_url)
    active_chat = get_active_chat()

    render_sidebar(api_url)
    st.title("DocuSense", anchor=False)
    st.caption("Private, local answers grounded in the documents you upload.")

    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                st.caption(f"Sources: {', '.join(message['sources'])}")

    selected_document_ids = st.session_state.selected_document_ids
    question = st.chat_input(
        "Ask about the selected documents" if selected_document_ids else "Select a document in the sidebar to begin",
        submit_mode="disable",
        disabled=not selected_document_ids,
    )
    if not question:
        return
    active_chat["messages"].append({"role": "user", "content": question})
    is_new_topic = active_chat["title"] == "New chat"
    if is_new_topic:
        active_chat["title"] = topic_title(question)
    with st.chat_message("user"):
        st.markdown(question)

    answer_text, sources = stream_answer(api_url, active_chat["id"], question, selected_document_ids)
    active_chat["messages"].append({"role": "assistant", "content": answer_text, "sources": sources})
    if is_new_topic:
        st.rerun()


def initialize_state() -> None:
    """Create per-browser chat state before rendering any stateful widgets."""
    st.session_state.setdefault("chats", [])
    st.session_state.setdefault("active_chat_id", None)
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("documents_loaded", False)
    st.session_state.setdefault("selected_document_ids", [])
    st.session_state.setdefault("delete_notice", None)
    if st.session_state.active_chat_id is None:
        create_new_chat()


def render_sidebar(api_url: str) -> None:
    """Render the collapsible native sidebar with uploads and chat topics."""
    with st.sidebar:
        st.header("DocuSense", anchor=False)
        delete_notice = st.session_state.pop("delete_notice")
        if delete_notice:
            st.toast(delete_notice, icon=":material/delete:")
        if st.button("New chat", icon=":material/add_comment:", width="stretch"):
            create_new_chat()
            st.rerun()

        history_slot = st.empty()
        render_history(history_slot)

        st.subheader("Documents")
        files = st.file_uploader(
            "Upload PDF, TXT, or Markdown files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="visible",
        )
        if st.button("Upload documents", icon=":material/upload_file:", disabled=not files, width="stretch"):
            try:
                result = upload_files(api_url, files)
                refresh_documents(api_url)
                new_document_ids = [str(document["id"]) for document in result["documents"]]
                if new_document_ids:
                    st.session_state.selected_document_ids = new_document_ids
                indexed_files = result["uploaded_files"]
                skipped_files = result["skipped_files"]
                if indexed_files:
                    st.toast(f"Indexed {len(indexed_files)} document(s).", icon=":material/check_circle:")
                if skipped_files:
                    st.info(f"Already indexed and skipped: {', '.join(skipped_files)}")
            except (HTTPError, URLError) as error:
                st.error(f"Upload failed: {format_error(error)}", icon=":material/error:")

        documents = st.session_state.documents
        valid_document_ids = {str(document["id"]) for document in documents}
        st.session_state.selected_document_ids = [
            document_id
            for document_id in st.session_state.selected_document_ids
            if document_id in valid_document_ids
        ]
        if documents:
            document_names = {str(document["id"]): str(document["name"]) for document in documents}
            st.multiselect(
                "Search in documents",
                options=list(document_names),
                format_func=document_names.__getitem__,
                key="selected_document_ids",
                placeholder="Choose one or more documents",
            )
            selected_document_ids = st.session_state.selected_document_ids
            if st.button(
                "Delete selected documents",
                icon=":material/delete:",
                disabled=not selected_document_ids,
                width="stretch",
            ):
                try:
                    deleted_documents = delete_documents(api_url, selected_document_ids)
                    refresh_documents(api_url)
                    st.session_state.delete_notice = (
                        f"Deleted {len(deleted_documents)} document(s) from local storage and the index."
                    )
                    st.rerun()
                except (HTTPError, URLError) as error:
                    st.error(f"Could not delete the selected documents: {format_error(error)}", icon=":material/error:")
        else:
            st.caption("Upload a document, then select it here before asking a question.")

        if st.button("Refresh document list", icon=":material/refresh:", width="stretch"):
            refresh_documents(api_url)
            st.rerun()


def render_history(history_slot: object) -> None:
    """Show only conversations that contain a user prompt."""
    chats_with_history = [chat for chat in st.session_state.chats if chat["messages"]]
    with history_slot.container():
        if not chats_with_history:
            return
        st.subheader("Chat history")
        for chat in reversed(chats_with_history):
            selected = chat["id"] == st.session_state.active_chat_id
            if st.button(
                chat["title"],
                key=f"chat-{chat['id']}",
                icon=":material/chat:" if not selected else ":material/forum:",
                type="primary" if selected else "secondary",
                width="stretch",
            ):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()


def stream_answer(
    api_url: str,
    chat_id: str,
    question: str,
    document_ids: list[str],
) -> tuple[str, list[str]]:
    """Show a thinking loader, then replace it with streamed answer text."""
    payload: dict[str, object] = {
        "session": f"streamlit-{chat_id}",
        "question": question,
        "uploaded_only": True,
        "document_ids": document_ids,
        "hybrid": True,
    }
    answer_text = ""
    sources: list[str] = []
    with st.chat_message("assistant"):
        thinking_slot = st.empty()
        with thinking_slot:
            st.status("Thinking...", state="running", expanded=False, type="compact")
        placeholder = st.empty()
        try:
            for event, data in stream_chat(api_url, payload):
                if event == "token":
                    if not answer_text:
                        thinking_slot.empty()
                    answer_text += str(data["text"])
                    placeholder.markdown(answer_text)
                elif event == "error":
                    thinking_slot.empty()
                    answer_text = str(data["message"])
                    placeholder.error(answer_text)
                elif event == "sources":
                    sources = display_source_names(data.get("sources", []))
        except HTTPError as error:
            thinking_slot.empty()
            answer_text = f"Unable to answer your question: {format_error(error)}"
            placeholder.error(answer_text)
        except URLError:
            thinking_slot.empty()
            answer_text = "Cannot reach the application server. Start the FastAPI server and try again."
            placeholder.error(answer_text)
        if not answer_text:
            thinking_slot.empty()
            answer_text = "No answer was returned. Please try again."
            placeholder.error(answer_text)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")
    return answer_text, sources


def create_new_chat() -> None:
    """Append a fresh conversation and make it the active local topic."""
    st.session_state.chats = [chat for chat in st.session_state.chats if chat["messages"]]
    chat = {"id": uuid4().hex, "title": "New chat", "messages": []}
    st.session_state.chats.append(chat)
    st.session_state.active_chat_id = chat["id"]


def get_active_chat() -> dict[str, object]:
    """Return the selected chat, restoring a fresh one if its state is stale."""
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    create_new_chat()
    return st.session_state.chats[-1]


def topic_title(question: str) -> str:
    """Create a short ChatGPT-style sidebar topic from the first user question."""
    normalized = " ".join(question.split())
    return normalized[:42] + ("…" if len(normalized) > 42 else "")


def refresh_documents(api_url: str) -> bool:
    """Load the current indexed upload list without exposing storage paths."""
    try:
        st.session_state.documents = list_documents(api_url)
        st.session_state.documents_loaded = True
        return True
    except (HTTPError, URLError):
        return False


def list_documents(api_url: str) -> list[dict[str, str]]:
    """Fetch the document IDs and display names available for selection."""
    request = Request(f"{api_url.rstrip('/')}/documents", method="GET")
    with urlopen(request, timeout=10) as response:  # noqa: S310 -- local configurable API endpoint.
        return list(json.loads(response.read().decode("utf-8")))


def upload_files(api_url: str, files: list[object]) -> dict[str, object]:
    """Send multiple uploads as general documents to the FastAPI multipart endpoint."""
    boundary = f"----RagUpload{os.urandom(12).hex()}"
    body_parts: list[bytes] = []
    body_parts.extend(_multipart_field(boundary, "document_type", "other"))
    for file in files:
        body_parts.extend(_multipart_file(boundary, file.name, file.getvalue()))
    body_parts.append(f"--{boundary}--\r\n".encode())
    request = Request(
        f"{api_url.rstrip('/')}/documents",
        data=b"".join(body_parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=600) as response:  # noqa: S310 -- local configurable API endpoint.
        return dict(json.loads(response.read().decode("utf-8")))


def delete_documents(api_url: str, document_ids: list[str]) -> list[dict[str, str]]:
    """Delete selected uploads from the API, vector index, and local storage."""
    deleted_documents: list[dict[str, str]] = []
    for document_id in document_ids:
        request = Request(f"{api_url.rstrip('/')}/documents/{document_id}", method="DELETE")
        with urlopen(request, timeout=30) as response:  # noqa: S310 -- local configurable API endpoint.
            deleted_documents.append(dict(json.loads(response.read().decode("utf-8"))))
    return deleted_documents


def display_source_names(sources: object) -> list[str]:
    """Show original filenames, never the local paths used by the backend."""
    names: list[str] = []
    for source in sources if isinstance(sources, list) else []:
        filename = os.path.basename(str(source))
        prefix, separator, original_name = filename.partition("_")
        is_upload_prefix = len(prefix) == 32 and all(character in "0123456789abcdef" for character in prefix.lower())
        name = original_name if separator and is_upload_prefix else filename
        if name and name not in names:
            names.append(name)
    return names


def format_error(error: HTTPError | URLError) -> str:
    """Give a concise UI-safe error without exposing backend internals."""
    if isinstance(error, HTTPError):
        return f"request failed with status {error.code}"
    return "cannot reach the application server"


def _multipart_field(boundary: str, name: str, value: str) -> list[bytes]:
    """Encode one plain multipart form field without a third-party HTTP client."""
    return [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
        value.encode(),
        b"\r\n",
    ]


def _multipart_file(boundary: str, filename: str, content: bytes) -> list[bytes]:
    """Encode one uploaded file while stripping path components from its name."""
    safe_filename = os.path.basename(filename).replace('"', "")
    return [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="files"; filename="{safe_filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        content,
        b"\r\n",
    ]


if __name__ == "__main__":
    main()
