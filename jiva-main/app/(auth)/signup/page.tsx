import { signUpWithCredentials } from "@/lib/actions/auth.actions"
import SignUpForm from "@/components/form/signup-form"

interface SignUpPageProps {
  searchParams?: {
    callbackUrl?: string
  }
}

const SignUpPage = ({ searchParams }: SignUpPageProps) => {
  const callbackUrl = searchParams?.callbackUrl || "/"; // Default value if callbackUrl is undefined

  return (
    <div className="w-full">
      <SignUpForm
        callbackUrl={callbackUrl}
        signUpWithCredentials={signUpWithCredentials}
      />
    </div>
  )
}

export default SignUpPage
