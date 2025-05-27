"use client";

import { useEffect } from "react";

export default function useHandleOutsideClick(ref: React.RefObject<HTMLElement>, active: boolean, callback: () => void) {
  function handleClickOutside(event: any) {
    const openPopover = document.querySelector('div[role="dialog"]');
    if (openPopover) {
      return;
    }

    if (ref.current && !ref.current.contains(event.target as Node)) {
      callback();
    }
  }

  useEffect(() => {
    if (active) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [ref, callback, active]);
}
