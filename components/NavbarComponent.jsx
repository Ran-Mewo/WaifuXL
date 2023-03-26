import { GitHubSVG } from '@/components/SVGComponents'
import Link from 'next/link'
import NewsBox from './NewsBox'
import { Sidebar } from './SidebarComponent'
import { useAppStateStore } from '@/services/useState'

const NavbarComponent = ({ currentPage }) => {
  const about_style = currentPage === 'about' ? 'text-white' : 'text-black'
  const index_style = currentPage === 'index' ? 'text-white' : 'text-black'
  const donate_style = currentPage === 'donate' ? 'text-white' : 'text-black'

  const mobile = useAppStateStore((state) => state.mobile)
  return (
    <>
      <GitHubSVG className="absolute right-5 bottom-4 md:top-4 z-40" />
      <header className="flex flex-col items-center relative w-full bg-pink md:mb-10 z-1">
        <nav className="">
          <div className="container mx-auto md:py-4 pt-0 flex justify-between items-center gap-4">
            <Link href="/about" className={`text-3xl font-semibold ${about_style}`}>
              About
            </Link>
            <Link href="/" className={`text-4xl font-bold ${index_style}`}>
              WaifuXL
            </Link>
            <Link href="/donate" className={`text-3xl font-semibold ${donate_style}`}>
              Donate
            </Link>
          </div>
        </nav>
        {currentPage === 'index' && !mobile && (
          <div className="flex flex-initial flex-row absolute left-0 top-full">
            <NewsBox />
          </div>
        )}
      </header>
    </>
  )
}

export default NavbarComponent
