import Link from "next/link";

import { TopNav } from "@/components/top-nav";
import { UploadWorkspace } from "@/components/upload-workspace";
import { Button } from "@/components/ui/button";

export default function UploadPage() {
  return (
    <main className="min-h-screen bg-background">
      <TopNav />
      <div className="container-app space-y-6 pb-12 pt-2">
        <div className="flex flex-col gap-4 rounded-xl border border-border bg-card/50 p-4 shadow-sm sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-foreground">New patient scan</h1>
            <p className="text-sm text-muted-foreground">Upload an image, capture patient context, then run AI analysis.</p>
          </div>
          <Button variant="outline" asChild>
            <Link href="/">Back to Dashboard</Link>
          </Button>
        </div>
        <UploadWorkspace />
      </div>
    </main>
  );
}
