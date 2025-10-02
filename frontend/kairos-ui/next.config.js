/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/kairos/:path*',
        destination: 'http://localhost:8080/api/:path*', // Proxy to Kairos backend
      },
      {
        source: '/ws/:path*',
        destination: 'http://localhost:8080/ws/:path*', // WebSocket proxy
      },
    ]
  },
}

module.exports = nextConfig