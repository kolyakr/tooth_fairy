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

export default function SignUpPage() {
  const router = useRouter();
  const [userId, setUserId] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const id = userId.trim();
    if (!id) {
      toast.error("Choose a user ID.");
      return;
    }
    if (id !== confirm.trim()) {
      toast.error("User ID and confirmation do not match.");
      return;
    }
    setBusy(true);
    try {
      const { access_token } = await requestDevToken(id);
      setAuthSession(access_token, id);
      toast.success("Account ready — you are signed in");
      router.push("/");
      router.refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Sign-up failed");
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
            <CardTitle className="text-2xl">Sign up</CardTitle>
            <CardDescription>
              Pick a unique <strong className="text-foreground">User ID</strong> for this environment (e.g. your name
              or clinic role). You will use it to sign in and to tie persisted scans to your session.
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
                  placeholder="e.g. dr.smith"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirm">Confirm user ID</Label>
                <Input
                  id="confirm"
                  name="confirm"
                  autoComplete="username"
                  placeholder="Repeat user ID"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  disabled={busy}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                This build uses the dev token endpoint (no password vault yet). Production should replace this with your
                identity provider.
              </p>
            </CardContent>
            <CardFooter className="flex flex-col gap-3 sm:flex-row sm:justify-between">
              <Button type="submit" className="w-full sm:w-auto" disabled={busy}>
                {busy ? <Loader2 className="size-4 animate-spin" /> : null}
                {busy ? "Creating…" : "Create account"}
              </Button>
              <Button type="button" variant="ghost" asChild className="w-full sm:w-auto">
                <Link href="/sign-in">Already have an account?</Link>
              </Button>
            </CardFooter>
          </form>
        </Card>
      </div>
    </main>
  );
}
