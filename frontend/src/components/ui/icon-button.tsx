import * as React from "react";

import { Slot } from "@radix-ui/react-slot";
import { type VariantProps, cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

import { Icons } from "../common/icons";

const iconButtonVariants = cva(
  "k-icon-btn inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent text-gray-600 dark:bg-gray-900 dark:hover:bg-gray-500 dark:text-white",
        primary: "bg-primary text-primary-foreground hover:bg-primary/90",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "bg-transparent hover:bg-accent hover:text-accent-foreground text-gray-600",
        link: "text-primary underline-offset-4 hover:underline",
        danger: "bg-red-500/80 text-primary-foreground hover:bg-red-500",
        circle: "rounded-full p-1 bg-background hover:bg-accent",
      },
      size: {
        xs: "min-w-5 w-5 h-5 [&_svg]:h-4 [&_svg]:w-4",
        sm: "min-w-6 w-6 h-6 [&_svg]:h-4 [&_svg]:w-4",
        md: "min-w-8 w-8 h-8 [&_svg]:h-4 [&_svg]:w-4",
        default: "min-w-9 w-9 h-9 [&_svg]:h-5 [&_svg]:w-5",
        lg: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "outline",
      size: "default",
    },
  }
);

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof iconButtonVariants> {
  asChild?: boolean;
  loading?: boolean;
}

const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ className, variant, size, loading = false, asChild = false, children, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        type="button"
        className={cn(iconButtonVariants({ variant, size, className }))}
        disabled={props.disabled || loading}
        ref={ref}
        onClick={(e) => {
          e.stopPropagation();
          if (props.onClick) {
            props.onClick(e);
          }
        }}
        {...props}
      >
        {loading ? <Icons.ButtonLoadingSpinner /> : children}
      </Comp>
    );
  }
);
IconButton.displayName = "IconButton";

export { IconButton, iconButtonVariants };
