
import { Navigate } from "react-router-dom";

const Index = () => {
  // Redirect root to dashboard
  return <Navigate to="/dashboard" replace />;
};

export default Index;
