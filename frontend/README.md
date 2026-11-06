# GitContext frontend

The React and TypeScript chat interface for GitContext.

## Run locally

Install dependencies and start Vite:

```powershell
npm install
npm run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:8000` during development.
For a separately hosted API, copy `.env.example` to `.env.local` and set
`VITE_API_BASE_URL` to that API's URL.

## Checks

```powershell
npm run build
npm run lint
npm test
```

Conversation history and drafts are stored in the browser's local storage.
Each chat has its own conversation ID so follow-up questions stay in context.
