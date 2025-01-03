import './globals.css'
import { Inter } from 'next/font/google'
import Navbar from '@/components/navbar/Navbar'
import Footer from '@/components/footer/Footer'
import Provider from '@/SessionProvider'
import Map from '@/components/map/map'
import LandingPage from '@/components/mencard/menuecard'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Create Next App',
  description: 'Generated by create next app',
}

export default function RootLayout({ children, session }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Provider>
          <Navbar />
         
          {children}
          <Map/>
          <Footer />
        </Provider>
      </body>
    </html>
  )
}
