import SignInForm from '@/components/form/signin-form'

interface SignInPageProps {
  searchParams?: {
    callbackUrl?: string
  }
}

const SignInPage = ({ searchParams }: SignInPageProps) => {
  const callbackUrl = searchParams?.callbackUrl || "/"; // Provide default if undefined

  return (
    <div className="w-full">
      <SignInForm callbackUrl={callbackUrl} />
    </div>
  )
}

export default SignInPage
