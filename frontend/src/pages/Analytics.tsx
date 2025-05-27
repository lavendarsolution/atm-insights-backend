
import { BarChart, LineChart, PieChart } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { mockATMs } from "@/lib/mock-data";

import {
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line as RechartsLine,
  LineChart as RechartsLineChart,
  Pie,
  PieChart as RechartsPieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function Analytics() {
  // Cash level distribution data
  const cashLevels = [
    { name: "< 20%", value: mockATMs.filter(atm => atm.cashLevel < 20).length, color: "#EF4444" },
    { name: "20% - 50%", value: mockATMs.filter(atm => atm.cashLevel >= 20 && atm.cashLevel < 50).length, color: "#F59E0B" },
    { name: "50% - 80%", value: mockATMs.filter(atm => atm.cashLevel >= 50 && atm.cashLevel < 80).length, color: "#3B82F6" },
    { name: "> 80%", value: mockATMs.filter(atm => atm.cashLevel >= 80).length, color: "#10B981" },
  ];
  
  // Status distribution data
  const statusData = [
    { name: "Active", value: mockATMs.filter(atm => atm.status === "active").length, color: "#10B981" },
    { name: "Warning", value: mockATMs.filter(atm => atm.status === "warning").length, color: "#F59E0B" },
    { name: "Error", value: mockATMs.filter(atm => atm.status === "error").length, color: "#EF4444" },
    { name: "Inactive", value: mockATMs.filter(atm => atm.status === "inactive").length, color: "#6B7280" },
  ];
  
  // Location data
  const locationData = mockATMs.reduce((acc, atm) => {
    if (!acc[atm.location]) {
      acc[atm.location] = { location: atm.location, count: 0, active: 0, warning: 0, error: 0, inactive: 0 };
    }
    acc[atm.location].count += 1;
    acc[atm.location][atm.status] += 1;
    return acc;
  }, {} as Record<string, any>);
  
  const locationChartData = Object.values(locationData).slice(0, 10);
  
  // Weekly trends data (mocked)
  const weeklyData = [
    { name: "Mon", transactions: 420, errors: 8 },
    { name: "Tue", transactions: 380, errors: 12 },
    { name: "Wed", transactions: 510, errors: 5 },
    { name: "Thu", transactions: 470, errors: 10 },
    { name: "Fri", transactions: 590, errors: 15 },
    { name: "Sat", transactions: 750, errors: 7 },
    { name: "Sun", transactions: 400, errors: 4 },
  ];
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Analytics</h1>
        <p className="text-muted-foreground">
          Analyze and visualize your ATM network data
        </p>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-base font-normal">Total ATMs</CardTitle>
            <div className="h-4 w-4 bg-muted rounded-full" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{mockATMs.length}</div>
            <p className="text-xs text-muted-foreground">Across {Object.keys(locationData).length} locations</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-base font-normal">Operational Rate</CardTitle>
            <div className="h-4 w-4 bg-success rounded-full" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {((mockATMs.filter(atm => atm.status === "active").length / mockATMs.length) * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {mockATMs.filter(atm => atm.status === "active").length} active ATMs
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-base font-normal">Issue Rate</CardTitle>
            <div className="h-4 w-4 bg-error rounded-full" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {((mockATMs.filter(atm => atm.status === "error" || atm.status === "warning").length / mockATMs.length) * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {mockATMs.filter(atm => atm.status === "error" || atm.status === "warning").length} ATMs with issues
            </p>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>ATM Status Overview</CardTitle>
          <CardDescription>Distribution of ATMs by status and cash level</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="status">
            <TabsList className="mb-4">
              <TabsTrigger value="status" className="flex items-center gap-1">
                <PieChart className="h-4 w-4" />
                <span>Status</span>
              </TabsTrigger>
              <TabsTrigger value="cash" className="flex items-center gap-1">
                <PieChart className="h-4 w-4" />
                <span>Cash Level</span>
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="status">
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsPieChart>
                    <Pie
                      data={statusData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    >
                      {statusData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
            
            <TabsContent value="cash">
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsPieChart>
                    <Pie
                      data={cashLevels}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    >
                      {cashLevels.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>ATMs by Location</CardTitle>
          <CardDescription>Distribution and status of ATMs across locations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsBarChart
                data={locationChartData}
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 60,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="location" 
                  angle={-45} 
                  textAnchor="end" 
                  height={70}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="active" stackId="a" name="Active" fill="#10B981" />
                <Bar dataKey="warning" stackId="a" name="Warning" fill="#F59E0B" />
                <Bar dataKey="error" stackId="a" name="Error" fill="#EF4444" />
                <Bar dataKey="inactive" stackId="a" name="Inactive" fill="#6B7280" />
              </RechartsBarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Weekly Trends</CardTitle>
          <CardDescription>Transaction count and error rates</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsLineChart
                data={weeklyData}
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <RechartsLine
                  yAxisId="left"
                  type="monotone"
                  dataKey="transactions"
                  stroke="#3B82F6"
                  name="Transactions"
                />
                <RechartsLine
                  yAxisId="right"
                  type="monotone"
                  dataKey="errors"
                  stroke="#EF4444"
                  name="Errors"
                />
              </RechartsLineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
