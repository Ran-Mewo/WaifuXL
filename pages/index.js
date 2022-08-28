import NavbarComponent from '../components/NavbarComponent'
import TitleComponent from '../components/TitleComponent'
import Sidebar from '../components/SidebarComponent'
import ImageDisplay from '../components/ImageDisplayComponent'
import AnnouncementComponent from '../components/Announcement'
import Error from '../components/ErrorComponent'
import ModalComponent from '../components/ModalComponent'
import { useAppStateStore } from '../services/useState'
import { useWindowSize } from '../services/windowUtilities'
import { useEffect } from 'react'

export default function Main() {
  const size = useWindowSize()
  const setMobile = useAppStateStore((state) => state.setMobile)

  useEffect(() => {
    setMobile(size.width / size.height < 1.0)
  }, [size])

  return (
    <>
      <Error />
      <div
        style={{
          backgroundImage: 'url("images/bg.svg")',
          backgroundSize: 'cover',
          backgroundPositionX: 'right',
        }}
      >
        <Sidebar />
        <ModalComponent />
        <main className="flex-1">
          <AnnouncementComponent announcement="Safari performance will be worse than other browsers. If possible use a non-webkit based browser." />
          <div className="flex flex-col items-center h-screen w-screen relative">
            <NavbarComponent currentPage="index" />
            <div className="h-3/4 grow w-full">
              <ImageDisplay />
              <TitleComponent />
            </div>
          </div>
        </main>
      </div>
    </>
  )
}
