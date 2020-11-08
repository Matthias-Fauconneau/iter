//#![allow(incomplete_features)]
#![feature(associated_type_bounds, min_const_generics, associated_type_defaults, in_band_lifetimes, array_value_iter, unboxed_closures, maybe_uninit_uninit_array, maybe_uninit_extra)] //generic_associated_types
macro_rules! assert_eq { ($left:expr, $right:expr) => (std::assert!($left == $right, "{} = {:?}, {} = {:?}", stringify!($left), $left, stringify!($right), $right)) }

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

pub trait IntoExactSizeIterator : IntoIterator<IntoIter:std::iter::ExactSizeIterator> {}
impl<I:IntoIterator<IntoIter:std::iter::ExactSizeIterator>>  IntoExactSizeIterator for I {}

pub trait FromExactSizeIterator<T> { fn from_iter<I:IntoIterator<Item=T>+IntoExactSizeIterator>(into_iter: I) -> Self; }
impl<T, const N : usize> FromExactSizeIterator<T> for [T; N] {
	#[track_caller] fn from_iter<I:IntoIterator<Item=T>+IntoExactSizeIterator>(into_iter: I) -> Self {
		let mut iter = into_iter.into_iter();
		assert_eq!(iter.len(), N);
		let mut array : [std::mem::MaybeUninit<T>; N] = std::mem::MaybeUninit::uninit_array();
		for e in array.iter_mut() { e.write(iter.next().unwrap()); }
		let array_as_initialized = unsafe { std::ptr::read(&array as *const _ as *const [T; N]) }; //61956
		std::mem::forget(array);
		array_as_initialized
	}
}
pub trait ExactSizeIterator : std::iter::ExactSizeIterator { #[track_caller] fn collect<B: FromExactSizeIterator<Self::Item>>(self) -> B where Self:Sized { FromExactSizeIterator::from_iter(self) } }
impl<I:std::iter::ExactSizeIterator> ExactSizeIterator for I {}
#[track_caller] pub fn collect<I:std::iter::ExactSizeIterator,B:FromExactSizeIterator<I::Item>>(iter: I) -> B { ExactSizeIterator::collect(iter) }

pub trait IntoValueIterator {
	type IntoIter: Iterator;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter;
}
impl<T, const N: usize> IntoValueIterator for [T; N] {
	type IntoIter = std::array::IntoIter<T, N>;
	fn into_iter(self) -> Self::IntoIter { std::array::IntoIter::new(self) }
}

pub struct Chain<A,B>{a: Option<A>, b: Option<B>}
impl<A:Iterator, B:Iterator<Item=A::Item>> Iterator for Chain<A, B> {
	type Item = A::Item;
	fn next(&mut self) -> Option<Self::Item> { if let Some(ref mut a) = self.a { a.next().or_else(||{ self.a = None; None }) } else { None }.or_else(|| self.b.as_mut().map(|b| b.next()).flatten()) }
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.a.as_ref().zip(self.b.as_ref()).map(|(a, b)| {
				let (a_lower, a_upper) = a.size_hint();
				let (b_lower, b_upper) = b.size_hint();
				let lower = a_lower.saturating_add(b_lower);
				let upper = a_upper.zip(b_upper).map(|(a, b)| a.checked_add(b)).flatten();
				(lower, upper)
		}).or(self.a.as_ref().map(|a| a.size_hint())).or(self.b.as_ref().map(|b| b.size_hint())).unwrap_or((0, Some(0)))
	}
}
impl<A:std::iter::ExactSizeIterator, B:std::iter::ExactSizeIterator<Item=A::Item>> std::iter::ExactSizeIterator for Chain<A, B> {}

pub trait IntoChain<B:Sized> : Sized { type Output; fn chain(self, b: B) -> Self::Output; }
impl<A: IntoValueIterator, B: IntoValueIterator+Sized> IntoChain<B> for A {
	type Output = Chain<A::IntoIter, B::IntoIter>;
	fn chain(self, b: B) -> Self::Output { Chain{a: Some(self.into_iter()), b: Some(b.into_iter())} }
}

use std::{iter::{Sum, Product}, ops};

pub trait IntoSum<T> { fn sum(self) -> T; }
impl<I:IntoIterator, T:Sum<I::Item>> IntoSum<T> for I { fn sum(self) -> T { Iterator::sum(self.into_iter()) } }

pub trait IntoProduct<T> { fn product(self) -> T; }
impl<I:IntoIterator, T:Product<I::Item>> IntoProduct<T> for I { fn product(self) -> T { Iterator::product(self.into_iter()) } }

pub trait IntoEnumerate<I> { fn enumerate(self) -> std::iter::Enumerate<I>; }
impl<I:IntoIterator> IntoEnumerate<I::IntoIter> for I { fn enumerate(self) -> std::iter::Enumerate<I::IntoIter> { Iterator::enumerate(self.into_iter()) } }

pub trait IntoCopied : Sized { fn copied(self) -> std::iter::Copied<Self>; }
impl<T:Copy+'t, I:Iterator<Item=&'t T>> IntoCopied for I { fn copied(self) -> std::iter::Copied<Self> { Iterator::copied(self.into_iter()) } }

pub trait Sub<B=Self> { type Output; fn sub(self, b: B) -> Self::Output; }
impl<T: 't, const N: usize> Sub for &'t [T; N] where &'t T:ops::Sub {
	type Output = [<&'t T as ops::Sub>::Output; N];
	fn sub(self, b: &'t [T; N]) -> Self::Output { ExactSizeIterator::collect(Iterator::map(Iterator::zip(self.into_iter(), b.into_iter()), |(a,b)| a-b)) }
}

pub trait Dot<T> { type Output; fn dot(self, other: T) -> Self::Output; }
impl<A: IntoIterator<Item:ops::Mul<B::Item>,IntoIter:'t>, B:IntoIterator<Item:>> Dot<B> for A where <A::Item as ops::Mul<B::Item>>::Output:std::iter::Sum {
	type Output = <A::Item as ops::Mul<B::Item>>::Output; fn dot(self, b: B) -> Self::Output { Iterator::sum(Iterator::map(Iterator::zip(self.into_iter(), b.into_iter()), |(a,b)| a*b)) }
}

pub struct Map<I,F>{iter: I, f: F}
pub trait IntoMap : IntoIterator+Sized { fn map<B, F:Fn(Self::Item)->B>(self, f: F) -> Map<Self, F> { Map{iter: self, f} } }
impl<I:IntoIterator> IntoMap for I {}

impl<I:IntoIterator, F: Fn<(I::Item,)>> IntoIterator for Map<I, F> {
	type IntoIter = std::iter::Map::<I::IntoIter, F>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { Iterator::map(self.iter.into_iter(), self.f) }
}
impl<I:IntoIterator, F: Fn<(I::Item,)>> IntoValueIterator for Map<I, F> {
	type IntoIter = std::iter::Map::<I::IntoIter, F>;
	fn into_iter(self) -> Self::IntoIter { Iterator::map(self.iter.into_iter(), self.f) }
}

pub struct Zip<A,B>{pub a: A, pub b: B}
pub trait IntoZip : Sized { fn zip<B>(self, b: B) -> Zip<Self, B> { Zip{a: self, b} } }
impl<T> IntoZip for T {}
impl<A:IntoIterator, B:IntoIterator> IntoIterator for Zip<A, B> {
	type IntoIter = std::iter::Zip::<A::IntoIter,B::IntoIter>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { Iterator::zip(self.a.into_iter(), self.b.into_iter()) }
}

pub trait Vector<const N: usize> : Sized { 	type Item; }
impl<T:Vector<N>, const N: usize> Vector<N> for &'t T { type Item = &'t T::Item; }
impl<T, const N: usize> Vector<N> for [T; N] {	type Item = T; }
impl<V:Vector<N>, F:Fn<(V::Item,)>, const N: usize> Vector<N> for Map<V, F> { type Item = F::Output; }
impl<A:Vector<N>, B:Vector<N>, const N: usize> Vector<N> for Zip<A, B> {	type Item = (<A as Vector<N>>::Item, <B as Vector<N>>::Item); }

pub trait VectorCollect<const N: usize> : Vector<N> {
	fn collect(self) -> [Self::Item; N];
}
impl<V:Vector<N>+IntoIterator<Item=<Self as Vector<N>>::Item>+IntoExactSizeIterator, const N: usize> VectorCollect<N> for V {
	fn collect(self) -> [Self::Item; N] { FromExactSizeIterator::from_iter(self.into_iter()) }
}

struct ConstRange<const N: usize>(std::ops::Range<usize>);
impl<const N: usize> ConstRange<N> { fn new() -> Self { Self(0..N) } }
impl<const N: usize> IntoIterator for ConstRange<N> { type IntoIter = std::ops::Range<usize>; type Item = <Self::IntoIter as Iterator>::Item; fn into_iter(self) -> Self::IntoIter { self.0 } }
impl<const N: usize> Vector<N> for ConstRange<N> { type Item = <Self as IntoIterator>::Item; }
#[track_caller] pub fn generate<T, F:Fn(usize)->T, const N:usize>(f : F) -> [T; N] { VectorCollect::collect(IntoMap::map(ConstRange::new(), f)) }

#[macro_export] macro_rules! zip {
  ($first:expr $(,)*) => { $first };
	($first:expr, $second:expr $(,)*) => { {use $crate::IntoZip; $crate::zip!($first).zip($second)} };
	($first:expr $( , $rest:expr )* $(,)* ) => { {use $crate::IntoZip; $crate::IntoMap::map($crate::zip!($first) $( .zip($rest) )*,$crate::zip!(@closure a => (a) $( , $rest )*) ) } };
	( @closure $p:pat => $tup:expr ) => { |$p| $tup };
	( @closure $p:pat => ( $($tup:tt)* ) , $_iter:expr $( , $tail:expr )* ) => { $crate::zip!(@closure ($p, b) => ( $($tup)*, b ) $( , $tail )*) };
}
#[macro_export] macro_rules! map { ($($args:expr),*; |$($params:ident),*| $expr:expr) => { $crate::IntoMap::map($crate::zip!($($args,)*), |($($params),*)| $expr) }; }
#[macro_export] macro_rules! eval { ($($args:expr),*; |$($params:ident),*| $expr:expr) => { $crate::VectorCollect::collect($crate::map!($($args),*; |$($params),*| $expr)) }; }

pub mod r#box { pub fn collect<T>(iter: impl IntoIterator<Item=T>) -> Box<[T]> { iter.into_iter().collect() } }

use std::convert::TryInto;
//pub trait Prefix { type Output<const S: usize>; fn prefix<const S: usize>(self) -> Self::Output<S>; }
//impl<T:Copy, const N: usize> Prefix for &[T; N] { type Output<const S: usize> = [T; S]; fn prefix<const S: usize>(self) -> Self::Output<S> { (&self[..S]).try_into().unwrap() } } // Error finalizing incremental compilation
pub trait Suffix<T> { fn suffix<const S: usize>(&self) -> &[T; S]; }
impl<T, const N: usize> Suffix<T> for [T; N] { fn suffix<const S: usize>(&self) -> &[T; S] { (&self[N-S..]).try_into().unwrap() } }
