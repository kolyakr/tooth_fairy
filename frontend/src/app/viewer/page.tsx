import { Suspense } from "react";

import { TopNav } from "@/components/top-nav";
import { Skeleton } from "@/components/ui/skeleton";
import { ViewerHub } from "@/components/viewer-hub";

function ViewerHubFallback() {
  return (
    <main className="min-h-screen bg-background">
      <TopNav />
      <div className="container-app space-y-6 py-6">
        <Skeleton className="h-48 w-full rounded-xl" />
        <Skeleton className="h-64 w-full rounded-xl" />
      </div>
    </main>
  );
}

export default function ViewerLegacyPage() {
  return (
    <Suspense fallback={<ViewerHubFallback />}>
      <ViewerHub />
    </Suspense>
  );
}
