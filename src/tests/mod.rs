mod frame;
mod gpu;
mod model;
mod platform;
mod style;
mod support;
mod surface;
mod vello;

trait UnwrapOrPanicForTest<T> {
    #[track_caller]
    fn unwrap_or_panic_for_test(self, message: &str) -> T;
}

impl<T> UnwrapOrPanicForTest<T> for Option<T> {
    #[track_caller]
    fn unwrap_or_panic_for_test(self, message: &str) -> T {
        match self {
            Some(value) => value,
            None => panic!("{message}"),
        }
    }
}

impl<T, E> UnwrapOrPanicForTest<T> for std::result::Result<T, E>
where
    E: std::fmt::Debug,
{
    #[track_caller]
    fn unwrap_or_panic_for_test(self, message: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic!("{message}: {error:?}"),
        }
    }
}
