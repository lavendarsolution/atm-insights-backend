"use client";

import { useRef } from "react";

import { useApp } from "@/providers/AppProvider";
import useHandleOutsideClick from "@/hooks/useHandleOutsideClick";
import clsx from "clsx";
import {
  BarChart3,
  BellIcon,
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  ListIcon,
  SettingsIcon,
  Sun,
} from "lucide-react";

import { cn } from "@/lib/utils";

import { IconButton } from "@/components/ui/icon-button";
import { Link, useLocation } from "react-router-dom";

const menuItems = [
  { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { path: "/atms", label: "ATM Registry", icon: ListIcon },
  { path: "/analytics", label: "Analytics", icon: BarChart3 },
  { path: "/alerts", label: "Alerts", icon: BellIcon },
  { path: "/settings", label: "Settings", icon: SettingsIcon },
];

export function AppSidebar() {
  const location = useLocation();
  const [{ sidebarOpen }, { setSidebarOpen }] = useApp();

  const ref = useRef();

  useHandleOutsideClick(ref, true, () => {
    if (sidebarOpen) {
      setSidebarOpen(false);
    }
  });

  return (
    <div
      className={clsx(
        "absolute z-50 flex h-full w-64 flex-col border-r border-gray-700 bg-gray-900 transition-all duration-300 md:relative md:left-0",
        {
          "md:w-[65px]": !sidebarOpen,
          "md:w-64": sidebarOpen,
          "-left-64": !sidebarOpen,
          "left-0": sidebarOpen,
        }
      )}
      ref={ref}
    >
      <div className={cn("px-5 py-6")}>
        {!sidebarOpen ? (
          <Sun className="h-6 w-6 text-yellow-400" />
        ) : (
          <img src="/assets/images/logo-white.svg" width="w-full" />
        )}
      </div>
      <div
        className={clsx("px-4 py-3 text-sm font-medium text-gray-400", {
          hidden: !sidebarOpen,
        })}
      >
        NAVIGATION
      </div>
      <div className="flex-1 px-3">
        <nav className="space-y-2">
          {menuItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 font-medium text-gray-300 transition-all hover:bg-gray-800 hover:text-white",
                location.pathname.startsWith(item.path)
                  ? "bg-gray-800 text-white"
                  : "",
                !sidebarOpen && "justify-center px-2"
              )}
              title={!sidebarOpen ? item.label : undefined}
            >
              <item.icon className="h-5 w-5" />
              {sidebarOpen && item.label}
            </Link>
          ))}
        </nav>
      </div>
      <IconButton
        variant="ghost"
        size="md"
        className="absolute -right-4 top-12 z-10 hidden rounded-full border border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-800 hover:text-white md:flex"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        {!sidebarOpen ? (
          <ChevronRight className="h-4 w-4" />
        ) : (
          <ChevronLeft className="h-4 w-4" />
        )}
      </IconButton>
    </div>
  );
}
