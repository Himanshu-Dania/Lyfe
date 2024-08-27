import Link from "next/link";
import Image from "next/image";
import MainNav from "@/components/shared/main-nav";
import UserNav from "@/components/shared/user-nav";
import ModeToggle from "@/components/shared/mode-toggle";
import Logo from "../../public/assets/jiva-logo.png";

const Navbar = () => {
  return (
    <header className="w-full fixed z-10 top-0 bg-gray-100 dark:bg-gray-900 border-b border-gray-200">
      <nav className="h-16 px-16 flex items-center">
        <Link href="/">
          <Image
            src={Logo.src}
            width={120}
            height={120}
            objectFit="contain"
            alt="Picture of the author"
          />
        </Link>
        <div className="ml-auto flex items-center space-x-4">
          <ModeToggle />
          <UserNav />
        </div>
      </nav>
    </header>
  );
};

export default Navbar;
