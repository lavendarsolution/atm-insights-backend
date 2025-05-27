import { z } from "zod";

export const loginSchema = z.object({
  email: z.string().email({ message: "Invalid email address" }),
  password: z
    .string()
    .min(8, { message: "Password must be at least 8 characters long" }),
});

export type LoginFormValues = z.infer<typeof loginSchema>;

export const registerSchema = z
  .object({
    first_name: z
      .string()
      .min(2, { message: "First name must be at least 2 characters long" }),
    last_name: z
      .string()
      .min(2, { message: "Last name must be at least 2 characters long" }),
    email: z.string().email({ message: "Invalid email address" }),
    password: z
      .string()
      .min(8, { message: "Password must be at least 8 characters long" }),
    password_confirmation: z.string(),
    acceptTerms: z.boolean().refine((val) => val === true, {
      message: "You must accept the terms and conditions",
    }),
  })
  .refine((data) => data.password === data.password_confirmation, {
    message: "Passwords do not match",
    path: ["confirmed"],
  });

export type RegisterFormValues = z.infer<typeof registerSchema>;

export const userInputSchema = z.object({
  first_name: z
    .string()
    .min(2, { message: "First name must be at least 2 characters long" }),
  last_name: z
    .string()
    .min(2, { message: "Last name must be at least 2 characters long" }),
  email: z.string().email({ message: "Invalid email address" }),
  role: z.string(),
  phone: z
    .string()
    .regex(
      /^(\+1|1)?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$/,
      "Invalid phone number format"
    )
    .or(z.literal(""))
    .nullable()
    .optional(),

  address_line_1: z.string().optional().nullable(),
  address_city: z.string().optional().nullable(),
  address_state: z.string().optional().nullable(),
  address_zip: z.string().optional().nullable(),

  site_id: z.number().optional().nullable(),
});

export type UserInput = z.infer<typeof userInputSchema>;

export const userSchema = userInputSchema.extend({
  id: z.number(),

  created_at: z.string(),
  updated_at: z.string(),

  last_login_at: z.string(),
});

export type User = z.infer<typeof userSchema>;
