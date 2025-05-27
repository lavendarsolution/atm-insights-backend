
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Search,
  Filter,
  Download,
  Plus,
  MapPin,
  Building,
  Calendar,
  Edit,
  Eye,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { StatusBadge } from "@/components/StatusBadge";
import { ATM, ATMStatus } from "@/lib/types";
import { mockATMs } from "@/lib/mock-data";
import { ATMActionsDropdown } from "@/components/ATMActionsDropdown";

const ITEMS_PER_PAGE = 10;

export default function ATMList() {
  const navigate = useNavigate();
  const [atms, setAtms] = useState<ATM[]>([]);
  const [filteredAtms, setFilteredAtms] = useState<ATM[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [manufacturerFilter, setManufacturerFilter] = useState<string>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [isLoading, setIsLoading] = useState(true);

  // Load ATM data
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setAtms(mockATMs);
      setIsLoading(false);
    };
    
    loadData();
  }, []);

  // Apply filters
  useEffect(() => {
    if (!atms.length) return;
    
    let filtered = [...atms];
    
    // Search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(
        atm =>
          atm.atm_id.toLowerCase().includes(term) ||
          atm.name.toLowerCase().includes(term) ||
          atm.location_address.toLowerCase().includes(term) ||
          atm.model.toLowerCase().includes(term) ||
          atm.manufacturer.toLowerCase().includes(term)
      );
    }
    
    // Status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter(atm => atm.status === statusFilter);
    }
    
    // Manufacturer filter
    if (manufacturerFilter !== "all") {
      filtered = filtered.filter(atm => atm.manufacturer === manufacturerFilter);
    }
    
    setFilteredAtms(filtered);
    setCurrentPage(1);
  }, [atms, searchTerm, statusFilter, manufacturerFilter]);

  // Get unique manufacturers for filter
  const manufacturers = Array.from(new Set(atms.map(atm => atm.manufacturer)));

  // Pagination
  const totalPages = Math.ceil(filteredAtms.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const currentAtms = filteredAtms.slice(startIndex, endIndex);

  const handleRowClick = (atmId: string) => {
    navigate(`/atm/${atmId}`);
  };

  const handleAddATM = () => {
    navigate("/atm/add");
  };

  const handleExport = () => {
    console.log("Exporting ATM data...");
    // Implementation for exporting data
  };

  const handleDeleteATM = (atmId: string) => {
    console.log("Deleting ATM:", atmId);
    // Implementation for deleting ATM
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">ATM Registry</h1>
          <p className="text-muted-foreground">
            Comprehensive list of all ATMs in your network with live telemetry data
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex gap-2">
          <Button variant="outline" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button onClick={handleAddATM}>
            <Plus className="h-4 w-4 mr-2" />
            Add ATM
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total ATMs</CardTitle>
            <Building className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{atms.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Online</CardTitle>
            <div className="h-2 w-2 rounded-full bg-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {atms.filter(atm => atm.lastTelemetry?.status === "online").length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Need Attention</CardTitle>
            <div className="h-2 w-2 rounded-full bg-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {atms.filter(atm => atm.status === "warning" || atm.status === "error").length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Low Cash</CardTitle>
            <div className="h-2 w-2 rounded-full bg-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {atms.filter(atm => (atm.lastTelemetry?.cash_level_percent || 0) < 20).length}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>ATM Directory</CardTitle>
          <CardDescription>
            Search and filter through all registered ATMs with live telemetry data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-6">
            <div className="relative max-w-md">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search ATMs by ID, name, location..."
                className="pl-8"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="flex items-center gap-2">
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-[150px]">
                  <SelectValue placeholder="All Statuses" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Statuses</SelectItem>
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                  <SelectItem value="inactive">Inactive</SelectItem>
                </SelectContent>
              </Select>
              <Select value={manufacturerFilter} onValueChange={setManufacturerFilter}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="All Manufacturers" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Manufacturers</SelectItem>
                  {manufacturers.map((manufacturer) => (
                    <SelectItem key={manufacturer} value={manufacturer}>
                      {manufacturer}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* ATM Table */}
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ATM ID</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Location</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Cash Level</TableHead>
                  <TableHead>Temperature</TableHead>
                  <TableHead>Network</TableHead>
                  <TableHead>Last Update</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={10} className="h-24 text-center">
                      <div className="flex flex-col items-center justify-center text-muted-foreground">
                        <Building className="h-8 w-8 mb-2 animate-pulse" />
                        Loading ATM data...
                      </div>
                    </TableCell>
                  </TableRow>
                ) : currentAtms.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={10} className="h-24 text-center">
                      No ATMs match your filters
                    </TableCell>
                  </TableRow>
                ) : (
                  currentAtms.map((atm) => (
                    <TableRow 
                      key={atm.atm_id}
                      className="cursor-pointer hover:bg-muted/50"
                      onClick={() => handleRowClick(atm.atm_id)}
                    >
                      <TableCell className="font-medium">{atm.atm_id}</TableCell>
                      <TableCell>{atm.name}</TableCell>
                      <TableCell className="max-w-xs truncate">{atm.location_address}</TableCell>
                      <TableCell>{atm.model}</TableCell>
                      <TableCell>
                        <StatusBadge status={atm.status} />
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span>{atm.lastTelemetry?.cash_level_percent || 0}%</span>
                          <div className={`h-2 w-8 rounded-full ${
                            (atm.lastTelemetry?.cash_level_percent || 0) < 20 
                              ? 'bg-red-500' 
                              : (atm.lastTelemetry?.cash_level_percent || 0) < 40 
                              ? 'bg-yellow-500' 
                              : 'bg-green-500'
                          }`} />
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className={
                          (atm.lastTelemetry?.temperature_celsius || 0) > 35 
                            ? 'text-red-600' 
                            : (atm.lastTelemetry?.temperature_celsius || 0) > 30 
                            ? 'text-yellow-600' 
                            : 'text-green-600'
                        }>
                          {atm.lastTelemetry?.temperature_celsius || 0}°C
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          atm.lastTelemetry?.network_status === 'connected' 
                            ? 'bg-green-100 text-green-700' 
                            : atm.lastTelemetry?.network_status === 'unstable' 
                            ? 'bg-yellow-100 text-yellow-700' 
                            : 'bg-red-100 text-red-700'
                        }`}>
                          {atm.lastTelemetry?.network_status || 'unknown'}
                        </span>
                      </TableCell>
                      <TableCell>
                        {new Date(atm.updated_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell className="text-right">
                        <div 
                          className="flex items-center justify-end"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ATMActionsDropdown 
                            atmId={atm.atm_id} 
                            onDelete={handleDeleteATM}
                          />
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-4">
              <Pagination>
                <PaginationContent>
                  <PaginationItem>
                    <PaginationPrevious 
                      onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                      className={currentPage === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                    />
                  </PaginationItem>
                  {[...Array(totalPages)].map((_, i) => (
                    <PaginationItem key={i + 1}>
                      <PaginationLink
                        onClick={() => setCurrentPage(i + 1)}
                        isActive={currentPage === i + 1}
                        className="cursor-pointer"
                      >
                        {i + 1}
                      </PaginationLink>
                    </PaginationItem>
                  ))}
                  <PaginationItem>
                    <PaginationNext 
                      onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                      className={currentPage === totalPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
                    />
                  </PaginationItem>
                </PaginationContent>
              </Pagination>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
