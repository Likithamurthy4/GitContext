import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState } from "react";
import "./App.css";

type Source = {
  source_id: string;
  file_path: string;
  start_line: number;
  end_line: number;
};

type AskResponse = {
  question: string;
  answer: string;
  sources: Source[];
};

function App() {
  const [question, setQuestion] = useState("");
  const [repoUrl, setRepoUrl] = useState(
    "https://github.com/pallets/itsdangerous"
  );
  const [result, setResult] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function askQuestion() {
    if (question.trim().length < 3 || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), repo_url: repoUrl.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : "Gemini API quota exceeded. Please try again after your quota resets."
          );
        }

        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Request failed (${response.status}).`
        );
      }


      if (typeof data.answer !== "string") {
        throw new Error("The API response did not contain an answer.");
      }

      setResult(data as AskResponse);

    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">G</span>
          <span>GitContext</span>
        </a>
        <span className="status">
          <span className="status-dot" />
          Code intelligence
        </span>
      </header>

      <section className="hero">
        <div className="eyebrow">
          <span>✳</span> YOUR AI CODEBASE COMPANION
        </div>
        <h1>
          Understand any codebase.
          <br />
          <span>Without the digging.</span>
        </h1>
        <p className="subtitle">
          Ask questions about your code. Get grounded answers with source
          references, powered by semantic search and AI.
        </p>

        <div className="search-card">
        <div className="repo-input">
          <label htmlFor="repo-url">GITHUB REPOSITORY</label>
          <input
            id="repo-url"
            type="url"
            value={repoUrl}
            onChange={(event) => setRepoUrl(event.target.value)}
            placeholder="https://github.com/owner/repository"
          />
        </div>
          <label htmlFor="question">ASK YOUR REPOSITORY</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void askQuestion();
              }
            }}
            placeholder="How does authentication work in this codebase?"
            rows={3}
          />
          <div className="search-footer">
            <span>Enter to ask · Shift + Enter for a new line</span>
            <button
              onClick={() => void askQuestion()}
              disabled={loading || question.trim().length < 3}
            >
              {loading ? "Thinking..." : "Ask GitContext ↗"}
            </button>
          </div>
        </div>

        <div className="suggestions">
          <span>TRY ASKING</span>
          {[
            "How does the library generate a signature?",
            "Where is the main signer class?",
            "How are signing keys derived?",
          ].map((suggestion) => (
            <button
              className="suggestion"
              key={suggestion}
              onClick={() => setQuestion(suggestion)}
            >
              {suggestion} <span>↗</span>
            </button>
          ))}
        </div>
      </section>

      <section className="results" aria-live="polite">
        {loading && (
          <div className="message-card">
            <span className="loader" />
            <div>
              <strong>Tracing the codebase...</strong>
              <p>Searching, reranking, and generating an answer.</p>
            </div>
          </div>
        )}

        {error && <div className="error-card">{error}</div>}

        {result && (
          <article className="answer-card">
            <div className="answer-heading">
              <span className="answer-icon">✳</span>
              <div>
                <span className="section-label">GITCONTEXT ANSWER</span>
                <h2>{result.question}</h2>
              </div>
            </div>
            <div className="answer-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {result.answer}
              </ReactMarkdown>
            </div>

            <div className="sources-heading">
              <h3>Sources</h3>
              <span>{result.sources.length} references</span>
            </div>

            <div className="source-list">
              {result.sources.map((source) => (
                <div className="source-item" key={source.source_id}>
                  <span className="source-badge">{source.source_id}</span>
                  <div className="source-details">
                    <code>{source.file_path}</code>
                    <span>
                      Lines {source.start_line}–{source.end_line}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </article>
        )}
      </section>

      <footer className="footer">
        <span>GitContext</span>
        <span>Built for developers, grounded in code.</span>
      </footer>
    </main>
  );
}

export default App;