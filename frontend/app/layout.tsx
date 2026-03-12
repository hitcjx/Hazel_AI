import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "榛子 - 心理健康评估系统",
  description: "一个温暖、安全的心理咨询空间",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
