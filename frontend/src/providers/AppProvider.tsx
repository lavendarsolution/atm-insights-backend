import {
  ReactNode,
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from "react";

import LoadingOverlay from "@/components/LoadingOverlay";

import { User } from "../auth/schema";

type State = {
  isAppLoading: boolean;
  user: User;
  sidebarOpen: boolean;
};

type Actions = {
  setIsAppLoading: (value: boolean) => void;
  setSidebarOpen: (value: boolean) => void;
};

export const AppContext = createContext<[State, Actions]>([
  {
    isAppLoading: false,
    user: null,
    sidebarOpen: false,
  },
  {
    setIsAppLoading: () => {},
    setSidebarOpen: () => {},
  },
]);
AppContext.displayName = "AppContext";

interface AppProviderProps {
  children: ReactNode;
}

function reducer(
  state: State,
  { type, payload }: { type: string; payload: any }
) {
  return {
    ...state,
    [type]: payload,
  };
}

export const AppProvider = ({ children }: AppProviderProps) => {
  const [state, dispatch] = useReducer(reducer, {
    isAppLoading: false,
    user: null,
    sidebarOpen: false,
  });

  const setIsAppLoading = (value: boolean) => {
    dispatch({ type: "isAppLoading", payload: value });
  };

  const setSidebarOpen = (value: boolean) => {
    dispatch({ type: "sidebarOpen", payload: value });
  };

  const value = useMemo((): [State, Actions] => {
    return [
      state as State,
      {
        setIsAppLoading,
        setSidebarOpen,
      },
    ];
  }, [state]);

  return (
    <AppContext.Provider value={value}>
      {children}
      <LoadingOverlay isLoading={state.isAppLoading} />
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext<[State, Actions]>(AppContext);
  if (context === undefined) {
    throw new Error("useApp must be used within an AppProvider");
  }
  return context;
};
