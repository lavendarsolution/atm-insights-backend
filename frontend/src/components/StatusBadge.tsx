
import { cn } from "@/lib/utils";
import { ATMStatus } from "@/lib/types";

interface StatusBadgeProps {
  status: ATMStatus;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const getStatusClass = (status: ATMStatus) => {
    switch (status) {
      case "active":
        return "bg-success text-success-foreground";
      case "warning":
        return "bg-warning text-warning-foreground";
      case "error":
        return "bg-error text-error-foreground";
      case "inactive":
        return "bg-muted text-muted-foreground";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getStatusLabel = (status: ATMStatus) => {
    switch (status) {
      case "active":
        return "Active";
      case "warning":
        return "Warning";
      case "error":
        return "Error";
      case "inactive":
        return "Inactive";
      default:
        return "Unknown";
    }
  };

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
        getStatusClass(status),
        className
      )}
    >
      <span className="relative flex h-2 w-2 mr-1">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-current"></span>
        <span className="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
      </span>
      {getStatusLabel(status)}
    </div>
  );
}
