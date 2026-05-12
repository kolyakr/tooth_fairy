import Link from "next/link";

import { AnalysisStatusPanel } from "@/components/analysis-status";
import { TopNav } from "@/components/top-nav";
import { Button } from "@/components/ui/button";

export default async function ViewerByIdPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  return (
    <main className="h-screen overflow-hidden bg-background">
      <TopNav />
      <div className="container-app flex h-[calc(100vh-4rem)] min-h-0 flex-col gap-4 overflow-hidden pb-4 pt-2">
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" asChild size="sm">
            <Link href="/">Back to Dashboard</Link>
          </Button>
          <Button variant="outline" asChild size="sm">
            <Link href="/upload">New scan</Link>
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-hidden">
          <AnalysisStatusPanel analysisId={id} />
        </div>
      </div>
    </main>
  );
}
