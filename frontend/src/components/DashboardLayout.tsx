import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { AppHeader } from "@/components/AppHeader";
import { AppFooter } from "@/components/AppFooter";
import { Outlet, useLocation } from "react-router-dom";
import { AppProvider } from "@/providers/AppProvider";

export function DashboardLayout() {
  const location = useLocation();

  // Determine page title based on current path
  const getPageTitle = () => {
    const path = location.pathname;
    if (path.includes("dashboard")) return "Dashboard";
    if (path.includes("atms")) return "ATM Registry";
    if (path.includes("analytics")) return "Analytics";
    if (path.includes("alerts")) return "Alerts";
    if (path.includes("settings")) return "Settings";
    return "ATM Insights";
  };

  return (
    <AppProvider>
      <div className="flex h-screen">
        <AppSidebar />
        <div className="flex flex-1 flex-col">
          <AppHeader title={getPageTitle()} />
          <main className="relative flex-1 bg-gray-50">
            <Outlet />
          </main>
          <AppFooter />
        </div>
      </div>
    </AppProvider>
  );
}
