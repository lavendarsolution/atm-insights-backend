
import { Mail } from "lucide-react";

export function AppFooter() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-[#2a4d59] py-3 text-white border-t">
      <div className="px-4">
        <div className="flex flex-wrap items-center justify-between text-sm">
          <p className="text-gray-300">© {currentYear} ATM Insight. All rights reserved.</p>
          <div className="flex items-center space-x-4">
            <a href="#" className="text-gray-300 transition-colors hover:text-white">
              Privacy Policy
            </a>
            <div className="flex items-center text-gray-300">
              <Mail className="mr-1 h-4 w-4" />
              <a href="mailto:support@atminsight.com" className="transition-colors hover:text-white">
                support@atminsight.com
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
