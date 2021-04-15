#![allow(incomplete_features)]
#![feature(associated_type_bounds, const_generics, const_evaluatable_checked, associated_type_defaults, in_band_lifetimes, unboxed_closures, maybe_uninit_uninit_array, maybe_uninit_extra, maybe_uninit_slice, step_trait)]
#![recursion_limit="6"]

pub fn zip(a: impl std::iter::IntoIterator<Item=f64>, b: impl Fn(usize)->f64) -> impl Iterator<Item=(f64, f64)> { a.into_iter().enumerate().map(move |(i,a)| (a,b(i))) }
pub fn dot(iter: impl std::iter::IntoIterator<Item=(f64, f64)>) -> f64 { iter.into_iter().map(|(a,b)| a*b).sum() }

use std::convert::TryInto;
pub trait Prefix<T> { fn prefix<const S: usize>(&self) -> &[T; S]; }
impl<T, const N: usize> Prefix<T> for [T; N] { fn prefix<const S: usize>(&self) -> &[T; S] { (&self[..S]).try_into().unwrap() } }
pub trait Suffix<T> { fn suffix<const S: usize>(&self) -> &[T; S]; }
impl<T, const N: usize> Suffix<T> for [T; N] { fn suffix<const S: usize>(&self) -> &[T; S] { (&self[N-S..]).try_into().unwrap() } }

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

pub struct Chain<A,B>{a: Option<A>, b: Option<B>} // +impl ExactSizeIterator
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
impl<A, B: Sized> IntoChain<B> for A {
	type Output = Chain<A, B>;
	fn chain(self, b: B) -> Self::Output { Chain{a: Some(self), b: Some(b)} }
}

pub trait IntoIterator { // +impl &Box<[T]>, [T; N]
	type Item;
	type IntoIter: Iterator<Item = Self::Item>;
	fn into_iter(self) -> Self::IntoIter;
}
// impl IntoIterator for std::iter::IntoIterator !Iterator
/*impl<I:std::iter::IntoIterator> IntoIterator for I { // Conflicts with impl IntoIterator for foreign !std::iter::IntoIterator which may implement std::iter::IntoIterator later ([])
	type IntoIter = <Self as std::iter::IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
}*/
impl<T> IntoIterator for &'t [T] {
	type IntoIter = <Self as std::iter::IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
}
impl<T, const N: usize> IntoIterator for &'t [T; N] {
	type IntoIter = <Self as std::iter::IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	//fn into_iter(self) -> Self::IntoIter { IntoIter(self) }
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
}
impl<T, const N: usize> IntoIterator for &'t mut [T; N] {
	type IntoIter = <Self as std::iter::IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
}
impl<T> IntoIterator for &'t Box<[T]> {
	type IntoIter = std::slice::Iter<'t, T>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { self.iter() }
}
impl<T, const N: usize> IntoIterator for [T; N] {
	type IntoIter = std::array::IntoIter<T, N>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::array::IntoIter::new(self) }
}
impl<T> IntoIterator for Box<[T]> {
	type IntoIter = std::vec::IntoIter<T>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { self.into_vec().into_iter() }
}
impl<K, V> IntoIterator for std::collections::BTreeMap<K,V> {
	type IntoIter = <Self as std::iter::IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
}
impl<I, F> IntoIterator for std::iter::Map<I, F> where Self:std::iter::IntoIterator {
	type IntoIter = <Self as std::iter::IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
}
impl<I, F> IntoIterator for std::iter::Filter<I, F> where Self:std::iter::IntoIterator {
	type IntoIter = <Self as std::iter::IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
}
impl<I, St, F> IntoIterator for std::iter::Scan<I, St, F> where Self:std::iter::IntoIterator {
	type IntoIter = <Self as std::iter::IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
}

pub mod into;

//impl std::iter::IntoIterator for IntoIterator !Iterator
impl<T> std::iter::IntoIterator for into::Copied<T> where Self:IntoIterator {
	type IntoIter = <Self as IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { IntoIterator::into_iter(self) }
}
impl<A, B> std::iter::IntoIterator for into::Zip<A, B> where Self:IntoIterator {
	type IntoIter = <Self as IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { IntoIterator::into_iter(self) }
}
impl<I, F> std::iter::IntoIterator for into::Map<I, F> where Self:IntoIterator {
	type IntoIter = <Self as IntoIterator>::IntoIter;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { IntoIterator::into_iter(self) }
}

pub trait IntoExactSizeIterator : IntoIterator<IntoIter:std::iter::ExactSizeIterator> {}
impl<I:IntoIterator<IntoIter:std::iter::ExactSizeIterator>> IntoExactSizeIterator for I {}

unsafe fn array_new<T, const N: usize>(init: impl FnOnce(&mut [std::mem::MaybeUninit<T>; N])) -> [T; N] {
	let mut array : [std::mem::MaybeUninit<T>; N] = std::mem::MaybeUninit::uninit_array();
	init(&mut array);
	let array_as_initialized = std::ptr::read(&array as *const _ as *const [T; N]); //61956
	std::mem::forget(array);
	array_as_initialized
}

pub trait FromExactSizeIterator<T> { fn from_iter<I:IntoIterator<Item=T>+IntoExactSizeIterator>(into_iter: I) -> Self; }
impl<T, const N : usize> FromExactSizeIterator<T> for [T; N] {
	#[track_caller] fn from_iter<I:IntoIterator<Item=T>+IntoExactSizeIterator>(into_iter: I) -> Self {
		let mut iter = into_iter.into_iter();
		assert_eq!(iter.len(), N);
		unsafe { array_new(|array| for e in array.iter_mut() { e.write(iter.next().unwrap()); }) }
	}
}

pub trait FromIterator<T> { fn from_iter<I:IntoIterator<Item=T>>(into_iter: I) -> Self; }
impl<T, const N : usize> FromIterator<T> for [T; N] {
	#[track_caller] fn from_iter<I:IntoIterator<Item=T>>(into_iter: I) -> Self {
		let mut iter = into_iter.into_iter();
		unsafe { array_new(|array| for e in array.iter_mut() { e.write(iter.next().unwrap()); }) }
	}
}

#[track_caller]
pub fn from_iter<T, I: IntoIterator<Item=T>+IntoExactSizeIterator, const N: usize>(iter: I) -> [T; N] where [T; N]: FromExactSizeIterator<<I as IntoIterator>::Item> {
	FromExactSizeIterator::from_iter(iter)
}

#[track_caller]
pub fn from_iter_<T, I: IntoIterator<Item=T>, const N: usize>(iter: I) -> [T; N] where [T; N]: FromIterator<<I as IntoIterator>::Item> {
	FromIterator::from_iter(iter)
}

pub trait Concat { type Output; fn concat(self) -> Self::Output; }
impl<T, const M: usize, const N: usize> Concat for [[T; N]; M] where [T; M*N]: {
	type Output = [T; M*N];
	fn concat(self) -> Self::Output { unsafe { array_new(|array| for (chunk, row) in array.chunks_mut(M).zip(self.into_iter()) { std::ptr::copy_nonoverlapping(row.as_ptr(), std::mem::MaybeUninit::slice_as_mut_ptr(chunk), row.len()); }) } }
}

pub mod vec;

pub fn box_collect<T>(iter: impl std::iter::IntoIterator<Item=T>) -> Box<[T]> { iter.into_iter().collect() }
pub fn map<T, U>(iter: impl std::iter::IntoIterator<Item=T>, f: impl Fn(T)->U) -> Box<[U]> { iter.into_iter().map(f).collect() }
pub fn eval<T>(len: usize, f: impl Fn(usize)->T) -> Box<[T]> { map(0..len, f) }
