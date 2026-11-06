import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

function apiResponse(
  body: unknown,
  status = 200,
): { ok: boolean; status: number; text: () => Promise<string> } {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  };
}

function askForm(question: string) {
  fireEvent.change(
    screen.getByRole("textbox", {
      name: "Ask a question about this repository",
    }),
    { target: { value: question } },
  );
  fireEvent.click(screen.getByRole("button", { name: /Ask GitContext/ }));
}

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  vi.unstubAllGlobals();
});

describe("GitContext chat", () => {
  it("sends follow-ups with the same conversation ID and renders cited sources", async () => {
    const ids: ReturnType<typeof crypto.randomUUID>[] = [
      "00000000-0000-4000-8000-000000000001",
      "00000000-0000-4000-8000-000000000002",
      "00000000-0000-4000-8000-000000000003",
    ];
    let generatedId = 0;
    vi.spyOn(crypto, "randomUUID").mockImplementation(
      () => ids[generatedId++] ?? ids[2],
    );
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        apiResponse({
          answer: "Authentication is implemented in `auth.py`. [S1]",
          conversation_id: ids[0],
          sources: [
            {
              source_id: "S1",
              file_path: "src/auth.py",
              start_line: 1,
              end_line: 5,
            },
          ],
        }),
      )
      .mockResolvedValueOnce(
        apiResponse({
          answer: "It validates the provided token. [S1]",
          conversation_id: ids[0],
          sources: [
            {
              source_id: "S1",
              file_path: "src/auth.py",
              start_line: 7,
              end_line: 12,
            },
          ],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    askForm("How does authentication work?");
    expect(
      await screen.findByText(/Authentication is implemented/),
    ).toBeTruthy();

    const firstRequest = JSON.parse(
      fetchMock.mock.calls[0][1].body as string,
    );
    expect(firstRequest.conversation_id).toBe(ids[0]);

    askForm("What does it validate?");
    expect(await screen.findByText(/It validates the provided token/)).toBeTruthy();

    const secondRequest = JSON.parse(
      fetchMock.mock.calls[1][1].body as string,
    );
    expect(secondRequest.conversation_id).toBe(firstRequest.conversation_id);
    expect(secondRequest.question).toBe("What does it validate?");
    const sourceLinks = screen.getAllByRole("link", { name: "src/auth.py" });
    expect(sourceLinks.at(-1)?.getAttribute("href")).toBe(
      "https://github.com/pallets/itsdangerous/blob/HEAD/src/auth.py#L7-L12",
    );
    expect(screen.getAllByText("S1")).toHaveLength(2);
  });

  it("shows API validation errors and keeps the question available to retry", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        apiResponse(
          { detail: "Enter a valid public GitHub repository URL." },
          400,
        ),
      ),
    );
    render(<App />);

    fireEvent.change(screen.getByLabelText("GITHUB REPOSITORY"), {
      target: { value: "https://example.com/not-a-repository" },
    });
    askForm("Explain the project structure");

    expect(
      await screen.findByRole("alert"),
    ).toBeTruthy();
    expect(
      screen.getByText("Enter a valid public GitHub repository URL."),
    ).toBeTruthy();
    expect(
      screen.getByRole("textbox", {
        name: "Ask a question about this repository",
      }),
    ).toHaveProperty("value", "Explain the project structure");
  });

  it("shows a progress indicator while the API request is pending", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise(() => undefined)),
    );
    render(<App />);

    askForm("Where is the entry point?");

    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toContain(
        "Searching the repository and preparing an answer",
      );
    });
    expect(screen.getByRole("button", { name: /Thinking/ })).toBeTruthy();
  });

  it("creates a fresh conversation from the new chat action", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        apiResponse({
          answer: "This repository contains a package. [S1]",
          sources: [
            {
              source_id: "S1",
              file_path: "README.md",
              start_line: 1,
              end_line: 3,
            },
          ],
        }),
      );
    fetchMock.mockResolvedValueOnce(
      apiResponse({
        answer: "Its configuration lives in pyproject.toml. [S1]",
        sources: [
          {
            source_id: "S1",
            file_path: "pyproject.toml",
            start_line: 1,
            end_line: 5,
          },
        ],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    askForm("Describe this package");
    await screen.findByText(/This repository contains a package/);
    const firstRequest = JSON.parse(
      fetchMock.mock.calls[0][1].body as string,
    );

    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    askForm("How is it configured?");
    expect(
      await screen.findByText(/Its configuration lives in pyproject.toml/),
    ).toBeTruthy();

    const secondRequest = JSON.parse(
      fetchMock.mock.calls[1][1].body as string,
    );
    expect(secondRequest.conversation_id).not.toBe(firstRequest.conversation_id);
    expect(
      JSON.parse(window.localStorage.getItem("gitcontext.chats.v1") ?? "[]"),
    ).toHaveLength(2);
  });

  it("reports malformed citation data instead of hiding the sources", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        apiResponse({
          answer: "The answer is supported. [S1]",
          sources: [{ source_id: "S1", file_path: "", start_line: 0, end_line: 2 }],
        }),
      ),
    );
    render(<App />);

    askForm("How does the code work?");

    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(
      screen.getByText("The API response contained invalid source citations."),
    ).toBeTruthy();
  });
});
