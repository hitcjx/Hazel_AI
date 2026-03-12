export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    // 调用Python后端API
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages }),
    });

    if (!response.ok) {
      throw new Error(`Backend responded with ${response.status}`);
    }

    // 注意：这里需要处理Python后端的流式响应
    // 暂时返回简单响应
    const data = await response.json();

    return new Response(JSON.stringify(data), {
      headers: {
        'Content-Type': 'application/json',
      },
    });

  } catch (error) {
    console.error('Chat API error:', error);
    return new Response(
      JSON.stringify({ error: '抱歉，我遇到了一些问题。请稍后再试。' }),
      {
        status: 500,
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
  }
}
