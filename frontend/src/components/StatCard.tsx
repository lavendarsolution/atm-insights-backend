
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  title: string;
  value: number | string;
  icon: LucideIcon;
  description?: string;
  type?: "default" | "success" | "warning" | "error";
}

export function StatCard({ title, value, icon: Icon, description, type = "default" }: StatCardProps) {
  const getTypeClass = () => {
    switch (type) {
      case "success":
        return "bg-success/10 text-success border-success/20";
      case "warning":
        return "bg-warning/10 text-warning-foreground border-warning/20";
      case "error":
        return "bg-error/10 text-error border-error/20";
      default:
        return "bg-card text-card-foreground border-border";
    }
  };

  return (
    <Card className={getTypeClass()}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && <p className="text-xs text-muted-foreground mt-1">{description}</p>}
      </CardContent>
    </Card>
  );
}
