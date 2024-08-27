import Link from "next/link";
import { footerLinks } from "@/constants";

import { Code } from "lucide-react";

import React from "react";
import { FaInstagram } from "react-icons/fa";
import { FaTwitter } from "react-icons/fa";
import { FaLinkedin } from "react-icons/fa";
import { FaYoutube } from "react-icons/fa";

function Footer() {
  return (
    <>
      <div className="flex flex-col justify-center items-center text-center  p-5 bg-gray-50">
        <h1 className=" text-gray-800 font-semibold">
          ©2024 All rights reserved | Build with ❤ by @
          <span className="hover:text-blue-600 font-semibold cursor-pointer">
            TeamJiva{" "}
          </span>
        </h1>
      </div>
    </>
  );
}

export default Footer;
