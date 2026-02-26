import { createRouter } from '@tanstack/react-router'
import { routeTree } from './routeTree.gen'

function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-cream">
      <div className="text-center">
        <h1 className="text-4xl font-semibold text-text-primary">404</h1>
        <p className="mt-2 text-text-primary/60">Page not found</p>
        <a href="/" className="mt-4 inline-block text-terracotta hover:text-terracotta-hover">
          Go home
        </a>
      </div>
    </div>
  )
}

export function getRouter() {
  const router = createRouter({
    routeTree,
    scrollRestoration: true,
    defaultNotFoundComponent: NotFound,
  })

  return router
}

declare module '@tanstack/react-router' {
  interface Register {
    router: ReturnType<typeof getRouter>
  }
}
