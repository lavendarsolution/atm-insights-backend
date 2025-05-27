
import { MoreHorizontal, Eye, Edit, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";

interface ATMActionsDropdownProps {
  atmId: string;
  onDelete?: (atmId: string) => void;
}

export function ATMActionsDropdown({ atmId, onDelete }: ATMActionsDropdownProps) {
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleView = () => {
    navigate(`/atm/${atmId}`);
  };

  const handleEdit = () => {
    navigate(`/atm/${atmId}/edit`);
  };

  const handleDelete = () => {
    if (onDelete) {
      onDelete(atmId);
    } else {
      // Default delete behavior
      toast({
        title: "Delete ATM",
        description: `ATM ${atmId} would be deleted`,
      });
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm">
          <MoreHorizontal className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[160px]">
        <DropdownMenuItem onClick={handleView}>
          <Eye className="h-4 w-4 mr-2" />
          View Details
        </DropdownMenuItem>
        <DropdownMenuItem onClick={handleEdit}>
          <Edit className="h-4 w-4 mr-2" />
          Edit
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleDelete} className="text-destructive">
          <Trash2 className="h-4 w-4 mr-2" />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
