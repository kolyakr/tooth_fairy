import { toast as sonnerToast } from "sonner";

const base = (message: string) => sonnerToast(message);

export const toast = Object.assign(base, {
  success: sonnerToast.success,
  error: sonnerToast.error,
  info: sonnerToast.info,
});

export const toastSuccess = sonnerToast.success;
export const toastError = sonnerToast.error;
export const toastInfo = sonnerToast.info;
