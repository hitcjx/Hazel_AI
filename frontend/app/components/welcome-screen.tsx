'use client';

import { useState } from 'react';
import Image from 'next/image';

export function WelcomeScreen({ onStart }: { onStart: () => void }) {
  return (
    <div className="fixed inset-0 flex flex-col items-center justify-center bg-paper-surface z-[100]">
      {/* 内容容器 - 确保居中 */}
      <div className="flex flex-col items-center justify-center">
        {/* Logo */}
        <div className="mb-8 animate-breathe flex justify-center">
          <Image
            src="/logo.png"
            alt="榛子"
            width={120}
            height={120}
            priority
          />
        </div>

        {/* 导语 */}
        <p className="text-xl text-paper-text text-center max-w-md px-8 leading-extra-loose mb-12">
          你好，我是榛子。<br />
          接下来的时间，我会陪你聊聊。<br />
          这里很安全，你可以放心表达。
        </p>

        {/* 开始按钮 */}
        <button
          onClick={onStart}
          className="px-8 py-3 bg-paper-ai hover:bg-opacity-80 text-paper-text rounded-full font-medium transition-all duration-300 hover:scale-105 hover:shadow-morandi-md shadow-morandi"
        >
          开始
        </button>
      </div>
    </div>
  );
}
