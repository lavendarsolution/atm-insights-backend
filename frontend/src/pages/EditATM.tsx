
import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Save } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ATM, ATMStatus } from "@/lib/types";
import { mockATMs } from "@/lib/mock-data";
import { useToast } from "@/hooks/use-toast";

export default function EditATM() {
  const { atmId } = useParams<{ atmId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  const [formData, setFormData] = useState({
    atm_id: "",
    name: "",
    location_address: "",
    model: "",
    manufacturer: "",
    status: "active" as ATMStatus,
  });

  useEffect(() => {
    const loadATM = async () => {
      setIsLoading(true);
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const atm = mockATMs.find(a => a.atm_id === atmId);
        if (atm) {
          setFormData({
            atm_id: atm.atm_id,
            name: atm.name,
            location_address: atm.location_address,
            model: atm.model,
            manufacturer: atm.manufacturer,
            status: atm.status,
          });
        } else {
          toast({
            title: "Error",
            description: "ATM not found",
            variant: "destructive",
          });
          navigate("/atms");
        }
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to load ATM data",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    if (atmId) {
      loadATM();
    }
  }, [atmId, navigate, toast]);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.atm_id || !formData.name || !formData.location_address || !formData.model || !formData.manufacturer) {
      toast({
        title: "Validation Error",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast({
        title: "Success",
        description: "ATM has been updated successfully",
      });
      
      navigate("/atms");
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update ATM. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => navigate("/atms")}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to ATMs
          </Button>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Edit ATM</h1>
            <p className="text-muted-foreground">Loading...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" onClick={() => navigate("/atms")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to ATMs
        </Button>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Edit ATM</h1>
          <p className="text-muted-foreground">
            Update ATM information for {formData.atm_id}
          </p>
        </div>
      </div>

      <Card className="max-w-2xl">
        <CardHeader>
          <CardTitle>ATM Information</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="atm_id">ATM ID *</Label>
                <Input
                  id="atm_id"
                  value={formData.atm_id}
                  onChange={(e) => handleInputChange("atm_id", e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="name">Name *</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => handleInputChange("name", e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="location_address">Location Address *</Label>
              <Input
                id="location_address"
                value={formData.location_address}
                onChange={(e) => handleInputChange("location_address", e.target.value)}
                required
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="model">Model *</Label>
                <Input
                  id="model"
                  value={formData.model}
                  onChange={(e) => handleInputChange("model", e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="manufacturer">Manufacturer *</Label>
                <Input
                  id="manufacturer"
                  value={formData.manufacturer}
                  onChange={(e) => handleInputChange("manufacturer", e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="status">Status</Label>
              <Select value={formData.status} onValueChange={(value: ATMStatus) => handleInputChange("status", value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                  <SelectItem value="inactive">Inactive</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex gap-2 pt-4">
              <Button type="submit" disabled={isSubmitting}>
                <Save className="h-4 w-4 mr-2" />
                {isSubmitting ? "Updating..." : "Update ATM"}
              </Button>
              <Button type="button" variant="outline" onClick={() => navigate("/atms")}>
                Cancel
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
