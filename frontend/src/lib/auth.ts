/** Browser session for API Bearer auth (JWT from dev token or future IdP). */

export const AUTH_CHANGED_EVENT = "toothfairy-auth-changed";

const TOKEN_KEY = "toothfairy_access_token";
const USER_KEY = "toothfairy_user_id";

export function getStoredAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function getStoredUserId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(USER_KEY);
}

export function setAuthSession(accessToken: string, userId: string): void {
  window.localStorage.setItem(TOKEN_KEY, accessToken);
  window.localStorage.setItem(USER_KEY, userId);
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

export function clearAuthSession(): void {
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(USER_KEY);
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

export function isSignedIn(): boolean {
  return Boolean(getStoredAccessToken());
}
