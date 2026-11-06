import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

type Source = {
  source_id: string;
  file_path: string;
  start_line: number;
  end_line: number;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

type ChatSession = {
  id: string;
  title: string;
  repoUrl: string;
  updatedAt: string;
  draft: string;
  messages: ChatMessage[];
};

type AskResponse = {
  answer: string;
  conversation_id?: string;
  sources: Source[];
};

type LoadedChats = {
  sessions: ChatSession[];
  error: string;
};

const DEFAULT_REPO_URL = "https://github.com/pallets/itsdangerous";
const CHAT_STORAGE_KEY = "gitcontext.chats.v1";
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "/api"
).replace(/\/+$/, "");

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isSource(value: unknown): value is Source {
  return (
    isRecord(value) &&
    typeof value.source_id === "string" &&
    /^S\d+$/.test(value.source_id) &&
    typeof value.file_path === "string" &&
    value.file_path.length > 0 &&
    typeof value.start_line === "number" &&
    Number.isInteger(value.start_line) &&
    value.start_line > 0 &&
    typeof value.end_line === "number" &&
    Number.isInteger(value.end_line) &&
    value.end_line >= value.start_line
  );
}

function isChatMessage(value: unknown): value is ChatMessage {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    (value.role === "user" || value.role === "assistant") &&
    typeof value.content === "string" &&
    (value.sources === undefined ||
      (Array.isArray(value.sources) && value.sources.every(isSource)))
  );
}

function isChatSession(value: unknown): value is ChatSession {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.title === "string" &&
    typeof value.repoUrl === "string" &&
    typeof value.updatedAt === "string" &&
    typeof value.draft === "string" &&
    Array.isArray(value.messages) &&
    value.messages.every(isChatMessage)
  );
}

function loadChats(): LoadedChats {
  try {
    const stored = window.localStorage.getItem(CHAT_STORAGE_KEY);
    if (!stored) return { sessions: [], error: "" };

    const parsed: unknown = JSON.parse(stored);
    if (!Array.isArray(parsed) || !parsed.every(isChatSession)) {
      throw new Error("Saved chat history has an unsupported format.");
    }

    return {
      sessions: parsed.sort(
        (left, right) =>
          Date.parse(right.updatedAt) - Date.parse(left.updatedAt),
      ),
      error: "",
    };
  } catch (error) {
    console.warn("Could not load saved GitContext chats.", error);
    return {
      sessions: [],
      error: "Saved chat history could not be loaded in this browser.",
    };
  }
}

function createChat(repoUrl = DEFAULT_REPO_URL): ChatSession {
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    repoUrl,
    updatedAt: new Date().toISOString(),
    draft: "",
    messages: [],
  };
}

function getErrorMessage(data: unknown, status: number): string {
  if (isRecord(data) && typeof data.detail === "string") {
    return data.detail;
  }

  if (isRecord(data) && Array.isArray(data.detail)) {
    const messages = data.detail
      .filter(isRecord)
      .map((item) => item.msg)
      .filter((message): message is string => typeof message === "string");
    if (messages.length) return messages.join(" ");
  }

  if (status === 429) {
    return "The AI provider quota is temporarily exceeded. Please try again later.";
  }
  if (status === 503) {
    return "The AI service is temporarily unavailable. Please try again shortly.";
  }
  return `Request failed (${status}). Please try again.`;
}

function getGitHubSourceUrl(repoUrl: string, source: Source): string | null {
  try {
    const repository = new URL(repoUrl);
    const path = repository.pathname
      .replace(/\.git\/?$/, "")
      .split("/")
      .filter(Boolean);

    if (repository.hostname !== "github.com" || path.length !== 2) {
      return null;
    }

    const filePath = source.file_path
      .split("/")
      .map(encodeURIComponent)
      .join("/");
    const line = source.end_line > source.start_line
      ? `#L${source.start_line}-L${source.end_line}`
      : `#L${source.start_line}`;

    return `https://github.com/${path.map(encodeURIComponent).join("/")}/blob/HEAD/${filePath}${line}`;
  } catch {
    return null;
  }
}

function App() {
  const [loadedChats] = useState(loadChats);
  const [sessions, setSessions] = useState<ChatSession[]>(() =>
    loadedChats.sessions.length ? loadedChats.sessions : [createChat()],
  );
  const [activeChatId, setActiveChatId] = useState(
    () => sessions[0].id,
  );
  const [storageWarning] = useState(loadedChats.error);
  const [pendingChatId, setPendingChatId] = useState<string | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [error, setError] = useState<{ chatId: string; message: string } | null>(
    null,
  );

  const activeChat = useMemo(
    () => sessions.find((session) => session.id === activeChatId) ?? sessions[0],
    [activeChatId, sessions],
  );
  const recentSessions = useMemo(
    () =>
      [...sessions].sort(
        (left, right) =>
          Date.parse(right.updatedAt) - Date.parse(left.updatedAt),
      ),
    [sessions],
  );
  const loading = pendingChatId !== null;

  useEffect(() => {
    try {
      window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(sessions));
    } catch (saveError) {
      console.warn("Could not save GitContext chats.", saveError);
    }
  }, [sessions]);

  function updateChat(chatId: string, update: (chat: ChatSession) => ChatSession) {
    setSessions((current) =>
      current.map((chat) => (chat.id === chatId ? update(chat) : chat)),
    );
  }

  function startNewChat() {
    const chat = createChat(activeChat.repoUrl);
    setSessions((current) => [chat, ...current]);
    setActiveChatId(chat.id);
    setError(null);
  }

  function clearActiveChat() {
    if (loading) return;

    const clearedChat = createChat(activeChat.repoUrl);
    setSessions((current) => [
      clearedChat,
      ...current.filter((chat) => chat.id !== activeChat.id),
    ]);
    setActiveChatId(clearedChat.id);
    setError(null);
  }

  function clearHistory() {
    if (loading) return;

    const chat = createChat(activeChat.repoUrl);
    setSessions([chat]);
    setActiveChatId(chat.id);
    setError(null);
  }

  async function askQuestion() {
    const chat = activeChat;
    const question = chat.draft.trim();
    if (question.length < 3 || loading) return;

    setPendingChatId(chat.id);
    setPendingQuestion(question);
    setError(null);

    updateChat(chat.id, (current) => ({
      ...current,
      title:
        current.messages.length === 0 && current.title === "New conversation"
          ? question
          : current.title,
      draft: "",
      updatedAt: new Date().toISOString(),
    }));

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          repo_url: chat.repoUrl.trim(),
          conversation_id: chat.id,
        }),
      });

      const responseText = await response.text();
      let data: unknown;
      try {
        data = responseText ? JSON.parse(responseText) : {};
      } catch {
        data = { detail: responseText };
      }

      if (!response.ok) {
        throw new Error(getErrorMessage(data, response.status));
      }

      if (
        !isRecord(data) ||
        typeof data.answer !== "string" ||
        !data.answer.trim()
      ) {
        throw new Error("The API response did not contain an answer.");
      }

      if (!Array.isArray(data.sources) || !data.sources.every(isSource)) {
        throw new Error("The API response contained invalid source citations.");
      }

      const answer: AskResponse = {
        answer: data.answer,
        conversation_id:
          typeof data.conversation_id === "string"
            ? data.conversation_id
            : undefined,
        sources: data.sources,
      };

      updateChat(chat.id, (current) => ({
        ...current,
        id: answer.conversation_id ?? current.id,
        messages: [
          ...current.messages,
          {
            id: crypto.randomUUID(),
            role: "user",
            content: question,
          },
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: answer.answer,
            sources: answer.sources,
          },
        ],
        updatedAt: new Date().toISOString(),
      }));

      const returnedConversationId = answer.conversation_id;
      if (returnedConversationId && returnedConversationId !== chat.id) {
        setActiveChatId((current) =>
          current === chat.id ? returnedConversationId : current,
        );
      }
    } catch (requestError) {
      const message =
        requestError instanceof TypeError
          ? "Could not connect to GitContext. Check that the API is running and try again."
          : requestError instanceof Error
            ? requestError.message
            : "An unexpected error occurred. Please try again.";
      setError({ chatId: chat.id, message });
      updateChat(chat.id, (current) => ({
        ...current,
        draft: current.draft || question,
      }));
    } finally {
      setPendingChatId(null);
      setPendingQuestion("");
    }
  }

  const activeError = error?.chatId === activeChat.id ? error.message : "";

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="GitContext home">
          <span className="brand-mark">G</span>
          <span>GitContext</span>
        </a>
        <div className="topbar-actions">
          <span className="status">
            <span className="status-dot" />
            AI code intelligence
          </span>
          <button
            className="new-chat-button"
            onClick={startNewChat}
            type="button"
          >
            <span aria-hidden="true">＋</span> New chat
          </button>
        </div>
      </header>

      <div className="app-layout">
        <aside className="sidebar" aria-label="Chat history">
          <button
            className="sidebar-new-chat"
            onClick={startNewChat}
            type="button"
          >
            <span aria-hidden="true">＋</span> New conversation
          </button>
          <div className="history-heading">
            <span>RECENT CHATS</span>
            <button
              className="text-button"
              onClick={clearHistory}
              disabled={loading}
              type="button"
            >
              Clear all
            </button>
          </div>
          <nav className="history-list">
            {recentSessions.map((chat) => (
              <button
                className={`history-item${chat.id === activeChat.id ? " active" : ""}`}
                key={chat.id}
                onClick={() => {
                  setActiveChatId(chat.id);
                  setError(null);
                }}
                type="button"
              >
                <span className="history-item-icon" aria-hidden="true">◷</span>
                <span className="history-item-title">{chat.title}</span>
              </button>
            ))}
          </nav>
          <div className="sidebar-footer">
            <span className="sidebar-footer-mark">G</span>
            <div>
              <strong>GitContext</strong>
              <span>Answers grounded in code</span>
            </div>
          </div>
        </aside>

        <section className="workspace">
          <div className="workspace-header">
            <div className="repo-field">
              <label htmlFor="repo-url">GITHUB REPOSITORY</label>
              <input
                id="repo-url"
                type="url"
                value={activeChat.repoUrl}
                onChange={(event) => {
                  const nextRepoUrl = event.target.value;
                  if (nextRepoUrl !== activeChat.repoUrl && activeChat.messages.length) {
                    const chat = createChat(nextRepoUrl);
                    setSessions((current) => [chat, ...current]);
                    setActiveChatId(chat.id);
                  } else {
                    updateChat(activeChat.id, (chat) => ({
                      ...chat,
                      repoUrl: nextRepoUrl,
                      updatedAt: new Date().toISOString(),
                    }));
                  }
                  setError(null);
                }}
                placeholder="https://github.com/owner/repository"
                autoComplete="url"
                spellCheck={false}
                disabled={loading}
              />
            </div>
            <button
              className="clear-chat-button"
              onClick={clearActiveChat}
              disabled={loading || activeChat.messages.length === 0}
              type="button"
            >
              <span aria-hidden="true">⌫</span> Clear chat
            </button>
          </div>

          <section className="conversation" aria-label="Conversation">
            <div className="conversation-scroll" aria-live="polite">
              {activeChat.messages.length === 0 &&
              pendingChatId === activeChat.id ? (
                <div className="message-list">
                  <article className="chat-message user">
                    <div className="message-content">
                      <p>{pendingQuestion}</p>
                    </div>
                  </article>
                  <div className="thinking" role="status">
                    <span className="loader" />
                    <span>
                      Searching the repository and preparing an answer…
                    </span>
                  </div>
                </div>
              ) : activeChat.messages.length === 0 ? (
                <div className="welcome">
                  <div className="welcome-orb" aria-hidden="true">✳</div>
                  <span className="eyebrow">YOUR AI CODEBASE COMPANION</span>
                  <h1>
                    Understand any codebase.
                    <span> Without the digging.</span>
                  </h1>
                  <p>
                    Ask a question about this repository. Get a clear answer
                    grounded in code, with references you can verify.
                  </p>
                  <div className="suggestions">
                    <span>GET STARTED</span>
                    {[
                      "How is the project structured?",
                      "Where is the main entry point?",
                      "How does error handling work?",
                    ].map((suggestion) => (
                      <button
                        className="suggestion"
                        key={suggestion}
                        onClick={() =>
                          updateChat(activeChat.id, (chat) => ({
                            ...chat,
                            draft: suggestion,
                          }))
                        }
                        type="button"
                      >
                        {suggestion} <span aria-hidden="true">↗</span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="message-list">
                  {activeChat.messages.map((message) => (
                    <article
                      className={`chat-message ${message.role}`}
                      key={message.id}
                    >
                      {message.role === "assistant" && (
                        <span className="message-avatar" aria-hidden="true">G</span>
                      )}
                      <div className="message-content">
                        {message.role === "assistant" ? (
                          <>
                            <div className="assistant-label">GITCONTEXT</div>
                            <div className="markdown-body">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {message.content}
                              </ReactMarkdown>
                            </div>
                            {message.sources && (
                              <div className="sources">
                                <div className="sources-heading">
                                  <h2>Sources</h2>
                                  <span>{message.sources.length} references</span>
                                </div>
                                {message.sources.length > 0 ? (
                                  <div className="source-list">
                                    {message.sources.map((source) => {
                                      const sourceUrl = getGitHubSourceUrl(
                                        activeChat.repoUrl,
                                        source,
                                      );
                                      return (
                                        <div
                                          className="source-item"
                                          key={`${source.source_id}-${source.file_path}-${source.start_line}`}
                                        >
                                          <span className="source-badge">
                                            {source.source_id}
                                          </span>
                                          <div className="source-details">
                                            {sourceUrl ? (
                                              <a
                                                href={sourceUrl}
                                                target="_blank"
                                                rel="noreferrer"
                                              >
                                                {source.file_path}
                                              </a>
                                            ) : (
                                              <code>{source.file_path}</code>
                                            )}
                                            <span>
                                              Lines {source.start_line}–{source.end_line}
                                            </span>
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                ) : (
                                  <p className="no-sources">
                                    No source files were returned for this answer.
                                  </p>
                                )}
                              </div>
                            )}
                          </>
                        ) : (
                          <p>{message.content}</p>
                        )}
                      </div>
                    </article>
                  ))}
                  {pendingChatId === activeChat.id && (
                    <>
                      <article className="chat-message user">
                        <div className="message-content">
                          <p>{pendingQuestion}</p>
                        </div>
                      </article>
                      <div className="thinking" role="status">
                        <span className="loader" />
                        <span>Searching the repository and preparing an answer…</span>
                      </div>
                    </>
                  )}
                  {activeError && (
                    <div className="error-card" role="alert">
                      <span className="error-icon" aria-hidden="true">!</span>
                      <div>
                        <strong>We couldn’t answer that yet</strong>
                        <p>{activeError}</p>
                      </div>
                    </div>
                  )}
                </div>
              )}
              {activeChat.messages.length === 0 && activeError && (
                <div className="error-card" role="alert">
                  <span className="error-icon" aria-hidden="true">!</span>
                  <div>
                    <strong>We couldn’t answer that yet</strong>
                    <p>{activeError}</p>
                  </div>
                </div>
              )}
            </div>

            {storageWarning && (
              <p className="storage-warning" role="status">{storageWarning}</p>
            )}

            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault();
                void askQuestion();
              }}
            >
              <label className="sr-only" htmlFor="question">
                Ask a question about this repository
              </label>
              <textarea
                id="question"
                value={activeChat.draft}
                onChange={(event) =>
                  updateChat(activeChat.id, (chat) => ({
                    ...chat,
                    draft: event.target.value,
                  }))
                }
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void askQuestion();
                  }
                }}
                placeholder="Ask anything about this codebase…"
                rows={2}
                disabled={loading}
              />
              <div className="composer-footer">
                <span>Enter to ask · Shift + Enter for a new line</span>
                <button
                  type="submit"
                  disabled={loading || activeChat.draft.trim().length < 3}
                >
                  {pendingChatId === activeChat.id ? (
                    <>
                      <span className="button-spinner" /> Thinking…
                    </>
                  ) : (
                    <>Ask GitContext <span aria-hidden="true">↗</span></>
                  )}
                </button>
              </div>
            </form>
            <p className="disclaimer">
              AI-generated answers may be incomplete. Verify important details
              against the cited source code.
            </p>
          </section>
        </section>
      </div>
    </main>
  );
}

export default App;
