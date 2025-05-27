import clsx from "clsx";

type ILoadingOverlayProps = {
  isLoading: boolean;
};

export default function LoadingOverlay({ isLoading }: ILoadingOverlayProps) {
  return (
    isLoading && (
      <div className={"full-spinner"}>
        <div></div>
      </div>
    )
  );
}
