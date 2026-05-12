"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Activity, Loader2 } from "lucide-react";
import { toast } from "sonner";

import { TopNav } from "@/components/top-nav";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { requestDevToken } from "@/lib/api-client";
import { setAuthSession } from "@/lib/auth";

export default function SignInPage() {
  const router = useRouter();
  const [userId, setUserId] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const id = userId.trim();
    if (!id) {
      toast.error("Enter a user ID.");
      return;
    }
    setBusy(true);
    try {
      const { access_token } = await requestDevToken(id);
      setAuthSession(access_token, id);
      toast.success("Signed in");
      router.push("/");
      router.refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <TopNav />
      <div className="container-app flex min-h-[calc(100vh-4rem)] items-center justify-center py-10">
        <Card className="w-full max-w-md border-border/80 shadow-sm">
          <CardHeader className="space-y-1 text-center">
            <div className="mx-auto flex size-12 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <Activity className="size-6" aria-hidden />
            </div>
            <CardTitle className="text-2xl">Sign in</CardTitle>
            <CardDescription>
              Use the same <strong className="text-foreground">User ID</strong> you chose at sign-up. The API issues a
              short-lived token when dev login is enabled.
            </CardDescription>
          </CardHeader>
          <form onSubmit={onSubmit}>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="userId">User ID</Label>
                <Input
                  id="userId"
                  name="userId"
                  autoComplete="username"
                  placeholder="e.g. dr.smith or clinic-admin"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  disabled={busy}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Backend: set <code className="rounded bg-muted px-1 py-0.5">TOOTHFAIRY_AUTH_JWT_SECRET</code> and{" "}
                <code className="rounded bg-muted px-1 py-0.5">TOOTHFAIRY_AUTH_DEV_LOGIN_ENABLED=true</code>, then
                restart the API.
              </p>
            </CardContent>
            <CardFooter className="flex flex-col gap-3 sm:flex-row sm:justify-between">
              <Button type="submit" className="w-full sm:w-auto" disabled={busy}>
                {busy ? <Loader2 className="size-4 animate-spin" /> : null}
                {busy ? "Signing in…" : "Sign in"}
              </Button>
              <Button type="button" variant="ghost" asChild className="w-full sm:w-auto">
                <Link href="/sign-up">Create an account</Link>
              </Button>
            </CardFooter>
          </form>
        </Card>
      </div>
    </main>
  );
}
