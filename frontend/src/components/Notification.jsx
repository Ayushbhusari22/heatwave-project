import { useEffect } from 'react';

export const Notification = ({ message, type, onDismiss }) => {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 5000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  return (
    <div className={`notification ${type}`}>
      {message}
      <button onClick={onDismiss}>×</button>
    </div>
  );
};