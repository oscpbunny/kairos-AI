/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/kairos/:path*',
        destination: 'http://localhost:8000/api/:path*', // Proxy to Kairos backend
      },
      {
        source: '/ws/:path*',
        destination: 'http://localhost:8000/ws/:path*', // WebSocket proxy
      },
    ]
  },
}

module.exports = nextConfig