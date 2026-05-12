"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { Activity, LayoutDashboard, LogOut, ScanSearch, Settings, User } from "lucide-react";

import { NavigationProgress } from "@/components/loaders/navigation-progress";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { AUTH_CHANGED_EVENT, clearAuthSession, getStoredUserId, isSignedIn } from "@/lib/auth";

export function TopNav() {
  const router = useRouter();
  const [signedIn, setSignedIn] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);

  const refreshAuth = useCallback(() => {
    setSignedIn(isSignedIn());
    setUserId(getStoredUserId());
  }, []);

  useEffect(() => {
    refreshAuth();
    const onAuth = () => refreshAuth();
    const onStorage = (e: StorageEvent) => {
      if (e.key === "toothfairy_access_token" || e.key === "toothfairy_user_id") refreshAuth();
    };
    window.addEventListener(AUTH_CHANGED_EVENT, onAuth);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(AUTH_CHANGED_EVENT, onAuth);
      window.removeEventListener("storage", onStorage);
    };
  }, [refreshAuth]);

  function onSignOut() {
    clearAuthSession();
    router.refresh();
  }

  return (
    <>
      <NavigationProgress />
      <header className="sticky top-0 z-50 border-b border-border/80 bg-card/80 backdrop-blur-md">
        <div className="container-app flex flex-wrap items-center gap-3 py-3 md:gap-4">
          <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight text-foreground">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
              <Activity className="size-5" aria-hidden />
            </span>
            <span className="hidden sm:inline">ToothFairy</span>
          </Link>

          <Tooltip>
            <TooltipTrigger asChild>
              <div className="relative min-w-0 flex-1 max-md:w-full">
                <ScanSearch className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search Patient ID or Name…"
                  className="h-10 bg-background/80 pl-10"
                  aria-label="Search patients"
                />
              </div>
            </TooltipTrigger>
            <TooltipContent>Patient search (wire to API later)</TooltipContent>
          </Tooltip>

          <div className="ml-auto flex flex-wrap items-center gap-1 sm:gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" asChild>
                  <Link href="/" aria-label="Dashboard">
                    <LayoutDashboard className="size-4" />
                  </Link>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Dashboard</TooltipContent>
            </Tooltip>

            <ThemeToggle />

            {signedIn ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="gap-2">
                    <User className="size-4" />
                    <span className="hidden max-w-[10rem] truncate md:inline">{userId ?? "Signed in"}</span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuLabel>Account</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem disabled className="text-xs font-normal text-muted-foreground">
                    {userId ? `User ID: ${userId}` : "Signed in"}
                  </DropdownMenuItem>
                  <DropdownMenuItem disabled>
                    <Settings className="mr-2 size-4" />
                    Settings
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={onSignOut}>
                    <LogOut className="mr-2 size-4" />
                    Sign out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <div className="flex items-center gap-1">
                <Button variant="ghost" size="sm" asChild>
                  <Link href="/sign-in">Sign in</Link>
                </Button>
                <Button size="sm" asChild>
                  <Link href="/sign-up">Sign up</Link>
                </Button>
              </div>
            )}
          </div>
        </div>
      </header>
    </>
  );
}
