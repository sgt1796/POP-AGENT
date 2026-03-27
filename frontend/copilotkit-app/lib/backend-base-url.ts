export function stripTrailingSlash(value: string) {
  return value.replace(/\/+$/, "");
}

export function getBackendBaseUrl() {
  return stripTrailingSlash(
    process.env.POP_AGENT_API_BASE_URL ||
      process.env.NEXT_PUBLIC_POP_AGENT_API_BASE_URL ||
      "http://127.0.0.1:8000"
  );
}
