import { useState } from "react";
import { 
  AlertTriangle, 
  Bell, 
  CheckCircle, 
  Clock, 
  Search 
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Alert } from "@/lib/types";
import { mockAlerts } from "@/lib/mock-data";

export default function Alerts() {
  const [searchTerm, setSearchTerm] = useState("");
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);

  // Filter alerts based on search term
  const filteredAlerts = alerts.filter(
    (alert) =>
      alert.atmId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      alert.message.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Alerts</h1>
        <p className="text-muted-foreground">
          Monitor and manage alerts from your ATM network
        </p>
      </div>

      {/* Search Bar */}
      <Card>
        <CardHeader>
          <CardTitle>Alerts</CardTitle>
          <CardDescription>
            Search and manage alerts in your ATM network
          </CardDescription>
          <div className="relative w-full max-w-md">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search alerts..."
              className="pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>ATM ID</TableHead>
                  <TableHead>Alert</TableHead>
                  <TableHead>Severity</TableHead>
                  <TableHead className="text-right">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredAlerts.map((alert) => (
                  <TableRow key={alert.id}>
                    <TableCell>
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </TableCell>
                    <TableCell>{alert.atmId}</TableCell>
                    <TableCell>{alert.message}</TableCell>
                    <TableCell>
                      <div
                        className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                          alert.severity === "high"
                            ? "bg-error/20 text-error"
                            : alert.severity === "medium"
                            ? "bg-warning/20 text-warning-foreground"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {alert.severity.charAt(0).toUpperCase() +
                          alert.severity.slice(1)}
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <Button variant="ghost" size="sm">
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
