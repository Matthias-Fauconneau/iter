#![feature(min_const_generics, maybe_uninit_uninit_array, maybe_uninit_extra, once_cell)]

pub trait Single: Iterator+Sized { fn single(mut self) -> Option<Self::Item> { self.next().filter(|_| self.next().is_none()) } }
impl<I:Iterator> Single for I {}

pub struct PeekingTakeWhile<'t, I:Iterator, P> { iter: &'t mut std::iter::Peekable<I>, predicate: P }
impl<'t, I:Iterator, P: Fn(&<I as Iterator>::Item) -> bool> PeekingTakeWhile<'t, I, P> {
	fn peek(&mut self) -> Option<&I::Item> {
        let Self{iter, predicate} = self;
        iter.peek().filter(|x| predicate(*x))
    }
}
impl<'t, I:Iterator, P: Fn(&<I as Iterator>::Item) -> bool> Iterator for PeekingTakeWhile<'t, I, P> {
    type Item = <I as Iterator>::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.peek()?;
        self.iter.next()
    }
}

pub trait PeekableExt<'t, I:Iterator> : Iterator {
    #[must_use] fn peeking_take_while<P:Fn(&<Self as Iterator>::Item) -> bool>(&'t mut self, predicate: P) -> PeekingTakeWhile<'t, I, P>;
}
impl<'t, I:Iterator> PeekableExt<'t, I> for std::iter::Peekable<I> {
    fn peeking_take_while<P:Fn(&<Self as Iterator>::Item) -> bool>(&'t mut self, predicate: P) -> PeekingTakeWhile<I, P> { PeekingTakeWhile{iter: self, predicate} }
}

pub trait NthOrLast : Iterator {
	fn nth_or_last(&mut self, mut n: usize) -> Result<Self::Item, Option<Self::Item>> {
		let mut last = None;
		for x in self {
			if n == 0 { return Ok(x); }
			n -= 1;
			last = Some(x);
		}
		Err(last)
	}
}
impl<I:Iterator> NthOrLast for I {}

pub fn from_iter<T>(iter: impl IntoIterator<Item=T>) -> Box<[T]> { use std::iter::FromIterator; Box::<[T]>::from_iter(iter) }

pub mod array {
	pub trait FromIterator<T> { fn from_iter<I:std::iter::IntoIterator<Item=T>>(into_iter: I) -> Self; }
	impl<T, const N : usize> FromIterator<T> for [T; N] {
			#[track_caller] fn from_iter<I>(into_iter: I) -> Self where I:std::iter::IntoIterator<Item=T> {
					let mut array : [std::mem::MaybeUninit<T>; N] = std::mem::MaybeUninit::uninit_array();
					let mut iter = into_iter.into_iter();
					for e in array.iter_mut() { e.write(iter.next().unwrap()); } // panic on short iter
					let array_as_initialized = unsafe { std::ptr::read(&array as *const _ as *const [T; N]) };
					std::mem::forget(array);
					array_as_initialized
			}
	}
	pub trait Iterator : std::iter::Iterator { #[track_caller] fn collect<B: FromIterator<Self::Item>>(self) -> B where Self:Sized { FromIterator::from_iter(self) } }
	impl<I:std::iter::Iterator> Iterator for I {}
	pub fn from_iter<T, const N:usize>(iter: impl IntoIterator<Item=T>) -> [T; N] { <[T; N]>::from_iter(iter) }
	pub fn generate<T, F:Fn(usize)->T, const N:usize>(f : F) -> [T; N] { from_iter((0..N).map(f)) }
	pub fn map<T, U, const N: usize>(v: &[T; N], f: impl Fn(&T)->U) -> [U; N] { from_iter(v.iter().map(f)) }
}
