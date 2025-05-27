
import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Thermometer,
  RefreshCcw,
  AlertTriangle,
  DollarSign,
  Cpu,
  HardDrive,
  Wifi,
  Clock,
  Activity
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { StatusBadge } from "@/components/StatusBadge";
import { mockATMs, getATMTelemetry, generateATMTelemetry } from "@/lib/mock-data";
import { ATM, ATMTelemetry } from "@/lib/types";
import { Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ATMDetail() {
  const { atmId } = useParams<{ atmId: string }>();
  const navigate = useNavigate();
  const [atm, setAtm] = useState<ATM | null>(null);
  const [telemetryData, setTelemetryData] = useState<ATMTelemetry[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate API call
    const fetchATMDetails = async () => {
      setLoading(true);
      try {
        await new Promise(resolve => setTimeout(resolve, 800));
        const foundATM = mockATMs.find(a => a.atm_id === atmId);
        
        if (foundATM) {
          setAtm(foundATM);
          const telemetries = getATMTelemetry(foundATM.atm_id);
          setTelemetryData(telemetries);
          setChartData(generateATMTelemetry(foundATM));
        } else {
          navigate("/atms", { replace: true });
        }
      } catch (error) {
        console.error("Error fetching ATM details:", error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchATMDetails();
  }, [atmId, navigate]);

  if (loading) {
    return (
      <div className="h-full w-full flex items-center justify-center">
        <RefreshCcw className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!atm) {
    return (
      <div className="text-center p-8">
        <h2 className="text-2xl font-bold">ATM not found</h2>
        <p className="text-muted-foreground mb-4">The requested ATM could not be found.</p>
        <Button onClick={() => navigate("/atms")}>Return to ATM List</Button>
      </div>
    );
  }

  const lastTelemetry = atm.lastTelemetry;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Button 
          variant="ghost" 
          size="icon"
          onClick={() => navigate("/atms")}
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">{atm.atm_id}</h1>
          <p className="text-muted-foreground">
            {atm.name} • {atm.model}
          </p>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* ATM Information Card */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">ATM Information</CardTitle>
            <CardDescription>Basic information and configuration</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">ATM ID</div>
              <div>{atm.atm_id}</div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">Name</div>
              <div>{atm.name}</div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">Location</div>
              <div>{atm.location_address}</div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">Model</div>
              <div>{atm.model}</div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">Manufacturer</div>
              <div>{atm.manufacturer}</div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">Status</div>
              <StatusBadge status={atm.status} />
            </div>
            
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">Created At</div>
              <div>{new Date(atm.created_at).toLocaleString()}</div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">Last Updated</div>
              <div>{new Date(atm.updated_at).toLocaleString()}</div>
            </div>
          </CardContent>
        </Card>

        {/* Current Telemetry Status */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-lg">Current Status</CardTitle>
            <CardDescription>Latest telemetry data from the ATM</CardDescription>
          </CardHeader>
          <CardContent>
            {lastTelemetry ? (
              <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
                {/* Operational Status */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">Operational Status</span>
                    <Activity className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`text-lg font-bold ${
                      lastTelemetry.status === 'online' ? 'text-green-600' :
                      lastTelemetry.status === 'error' ? 'text-red-600' :
                      lastTelemetry.status === 'maintenance' ? 'text-yellow-600' : 'text-gray-600'
                    }`}>
                      {lastTelemetry.status}
                    </div>
                  </div>
                </div>

                {/* Cash Level */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">Cash Level</span>
                    <DollarSign className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="text-lg font-bold">{lastTelemetry.cash_level_percent || 0}%</div>
                  </div>
                  <div className="h-2 w-full rounded-full bg-muted">
                    <div
                      className={`h-full rounded-full ${
                        (lastTelemetry.cash_level_percent || 0) < 20
                          ? "bg-red-500"
                          : (lastTelemetry.cash_level_percent || 0) < 40
                          ? "bg-yellow-500"
                          : "bg-green-500"
                      }`}
                      style={{ width: `${lastTelemetry.cash_level_percent || 0}%` }}
                    ></div>
                  </div>
                </div>

                {/* Temperature */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">Temperature</span>
                    <Thermometer className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="text-lg font-bold">{lastTelemetry.temperature_celsius || 0}°C</div>
                  </div>
                </div>

                {/* CPU Usage */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">CPU Usage</span>
                    <Cpu className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="text-lg font-bold">{lastTelemetry.cpu_usage_percent || 0}%</div>
                </div>

                {/* Memory Usage */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">Memory Usage</span>
                    <HardDrive className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="text-lg font-bold">{lastTelemetry.memory_usage_percent || 0}%</div>
                </div>

                {/* Network Status */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">Network</span>
                    <Wifi className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="text-lg font-bold">{lastTelemetry.network_status || 'unknown'}</div>
                  {lastTelemetry.network_latency_ms && (
                    <div className="text-sm text-muted-foreground">{lastTelemetry.network_latency_ms}ms</div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground">No telemetry data available</div>
            )}
          </CardContent>
          <CardFooter>
            <Button variant="outline" className="w-full" size="sm">
              <RefreshCcw className="h-4 w-4 mr-2" />
              Refresh Data
            </Button>
          </CardFooter>
        </Card>
      </div>

      {/* Telemetry History */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Telemetry History</CardTitle>
          <CardDescription>Historical data and recent telemetry records</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="table">
            <TabsList className="mb-4">
              <TabsTrigger value="table">Recent Data</TabsTrigger>
              <TabsTrigger value="charts">Charts</TabsTrigger>
            </TabsList>
            
            <TabsContent value="table">
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Cash Level</TableHead>
                      <TableHead>Temperature</TableHead>
                      <TableHead>CPU</TableHead>
                      <TableHead>Memory</TableHead>
                      <TableHead>Network</TableHead>
                      <TableHead>Error</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {telemetryData.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={8} className="h-24 text-center">
                          No telemetry data available
                        </TableCell>
                      </TableRow>
                    ) : (
                      telemetryData.map((telemetry, index) => (
                        <TableRow key={index}>
                          <TableCell className="font-mono text-sm">
                            {new Date(telemetry.time).toLocaleString()}
                          </TableCell>
                          <TableCell>
                            <span className={`text-xs px-2 py-1 rounded-full ${
                              telemetry.status === 'online' ? 'bg-green-100 text-green-700' :
                              telemetry.status === 'error' ? 'bg-red-100 text-red-700' :
                              telemetry.status === 'maintenance' ? 'bg-yellow-100 text-yellow-700' :
                              'bg-gray-100 text-gray-700'
                            }`}>
                              {telemetry.status}
                            </span>
                          </TableCell>
                          <TableCell>{telemetry.cash_level_percent || 0}%</TableCell>
                          <TableCell>{telemetry.temperature_celsius || 0}°C</TableCell>
                          <TableCell>{telemetry.cpu_usage_percent || 0}%</TableCell>
                          <TableCell>{telemetry.memory_usage_percent || 0}%</TableCell>
                          <TableCell>
                            <span className={`text-xs ${
                              telemetry.network_status === 'connected' ? 'text-green-600' :
                              telemetry.network_status === 'unstable' ? 'text-yellow-600' :
                              'text-red-600'
                            }`}>
                              {telemetry.network_status || 'unknown'}
                            </span>
                          </TableCell>
                          <TableCell>
                            {telemetry.error_code && (
                              <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded">
                                {telemetry.error_code}
                              </span>
                            )}
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </div>
            </TabsContent>
            
            <TabsContent value="charts">
              <Tabs defaultValue="cash" className="w-full">
                <TabsList className="mb-4">
                  <TabsTrigger value="cash">Cash Level</TabsTrigger>
                  <TabsTrigger value="temperature">Temperature</TabsTrigger>
                  <TabsTrigger value="transactions">Transactions</TabsTrigger>
                </TabsList>
                
                <TabsContent value="cash" className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="cashLevel"
                        stroke="#3B82F6"
                        strokeWidth={2}
                        name="Cash Level (%)"
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
                
                <TabsContent value="temperature" className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 50]} />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="temperature"
                        stroke="#EF4444"
                        strokeWidth={2}
                        name="Temperature (°C)"
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
                
                <TabsContent value="transactions" className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="transactions"
                        stroke="#10B981"
                        strokeWidth={2}
                        name="Daily Transactions"
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
              </Tabs>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Error Information */}
      {lastTelemetry?.error_code && (
        <Card className="border-red-200">
          <CardHeader className="flex flex-row items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <div>
              <CardTitle className="text-lg">Error Information</CardTitle>
              <CardDescription>Current system errors and alerts</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span>Error Code:</span>
                <span className="font-bold text-red-600">{lastTelemetry.error_code}</span>
              </div>
              
              {lastTelemetry.error_message && (
                <div className="flex items-center justify-between">
                  <span>Error Message:</span>
                  <span className="font-medium">{lastTelemetry.error_message}</span>
                </div>
              )}
            </div>
          </CardContent>
          <CardFooter>
            <Button className="w-full" variant="outline">
              Schedule Maintenance
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
}
