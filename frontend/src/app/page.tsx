import { DashboardPageBody } from "@/components/dashboard-page-body";
import { TopNav } from "@/components/top-nav";

export default function DashboardPage() {
  return (
    <main className="min-h-screen bg-background">
      <TopNav />
      <DashboardPageBody />
    </main>
  );
}
