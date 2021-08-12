#![feature(trait_alias,associated_type_bounds,in_band_lifetimes,unboxed_closures)]
pub trait Prefix<T> { fn prefix<const S: usize>(&self) -> &[T; S]; }
impl<T, const N: usize> Prefix<T> for [T; N] { fn prefix<const S: usize>(&self) -> &[T; S] { self[..S].try_into().unwrap() } }
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

pub struct ChainIterator<A,B>{a: Option<A>, b: Option<B>} // +impl ExactSizeIterator
impl<A:Iterator, B:Iterator<Item=A::Item>> Iterator for ChainIterator<A, B> {
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
impl<A:ExactSizeIterator, B:ExactSizeIterator<Item=A::Item>> ExactSizeIterator for ChainIterator<A, B> {}
pub trait IntoChain : Sized { fn chain<B>(self, b: B) -> ChainIterator<Self, B> { ChainIterator{a: Some(self), b: Some(b)} } }
impl<A> IntoChain for A {}

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
impl<T, const N: usize> IntoIterator for [T; N] {
	type IntoIter = std::array::IntoIter<T, N>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::array::IntoIter::new(self) }
}
impl<T> IntoIterator for &'t Box<[T]> {
	type IntoIter = std::slice::Iter<'t, T>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { self.iter() }
}
impl<T> IntoIterator for &'t Vec<T> {
	type IntoIter = std::slice::Iter<'t, T>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { self.iter() }
}
impl<T> IntoIterator for Box<[T]> {
	type IntoIter = std::vec::IntoIter<T>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self.into_vec()) }
}
impl<T> IntoIterator for Vec<T> {
	type IntoIter = std::vec::IntoIter<T>;
	type Item = <Self::IntoIter as Iterator>::Item;
	fn into_iter(self) -> Self::IntoIter { std::iter::IntoIterator::into_iter(self) }
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
impl<A, B> IntoIterator for std::iter::Zip<A, B> where Self:std::iter::IntoIterator {
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

pub trait IntoExactSizeIterator = IntoIterator<IntoIter:ExactSizeIterator>;
//impl<I:IntoIterator<IntoIter:ExactSizeIterator>> IntoExactSizeIterator for I {}

pub trait FromExactSizeIterator<T> { fn from_iter<I:IntoIterator<Item=T>+IntoExactSizeIterator>(into_iter: I) -> Self; }
impl<T, const N : usize> FromExactSizeIterator<T> for [T; N] {
	#[track_caller] fn from_iter<I:IntoIterator<Item=T>+IntoExactSizeIterator>(into_iter: I) -> Self {
		let mut iter = into_iter.into_iter();
		assert_eq!(iter.len(), N);
		[(); N].map(|()| iter.next().unwrap())
	}
}

pub trait FromIterator<T> { fn from_iter<I:IntoIterator<Item=T>>(into_iter: I) -> Self; }
impl<T, const N : usize> FromIterator<T> for [T; N] {
	#[track_caller] fn from_iter<I:IntoIterator<Item=T>>(into_iter: I) -> Self {
		let mut iter = into_iter.into_iter();
		[(); N].map(|()| iter.next().unwrap())
	}
}

#[track_caller]
pub fn from_iter<T, I: IntoIterator<Item=T>+IntoExactSizeIterator, const N: usize>(iter: I) -> [T; N] where [T; N]: FromExactSizeIterator<<I as IntoIterator>::Item> {
	FromExactSizeIterator::from_iter(iter)
}

#[derive(Clone, Copy)] pub struct ConstRange<const N: usize>;
impl<const N: usize> IntoIterator for ConstRange<N> { type IntoIter = std::ops::Range<usize>; type Item = <Self::IntoIter as Iterator>::Item; fn into_iter(self) -> Self::IntoIter { 0..N } }

pub trait IntoConstSizeIterator<const N: usize> : IntoExactSizeIterator+Sized {
	fn collect(self) -> [<Self as IntoIterator>::Item; N] { FromExactSizeIterator::from_iter(self) }
	fn map<F:Fn<(<Self as IntoIterator>::Item,)>>(self, f: F) -> into::Map<Self, F> { into::IntoMap::map(self, f) } // without "use IntoMap".map (conflicts with Iterator)
}
impl<const N: usize> IntoConstSizeIterator<N> for ConstRange<N> {}
impl<T, const N: usize> IntoConstSizeIterator<N> for &[T; N] {}
impl<T, const N: usize> IntoConstSizeIterator<N> for [T; N] {}
impl<T:Copy+'t, I:IntoIterator<Item=&'t T>+IntoConstSizeIterator<N>, const N: usize> IntoConstSizeIterator<N> for into::Copied<I> {}
impl<I:IntoConstSizeIterator<N>, const N: usize> IntoConstSizeIterator<N> for into::Enumerate<I> {}
impl<I:IntoConstSizeIterator<N>, F:Fn<(<I as IntoIterator>::Item,)>, const N: usize> IntoConstSizeIterator<N> for into::Map<I, F> {}
impl<A:IntoConstSizeIterator<N>, B:IntoConstSizeIterator<N>, const N: usize> IntoConstSizeIterator<N> for into::Zip<A, B> {}

#[track_caller] pub fn generate<T, F:Fn(usize)->T, const N:usize>(f : F) -> into::Map<ConstRange<N>, F> { IntoConstSizeIterator::map(ConstRange, f) }

//pub fn eval<T, U, const N: usize>(v: impl IntoConstSizeIterator<N>+IntoIterator<Item=T,IntoIter:ExactSizeIterator>, f: impl Fn(T)->U) -> [U; N] { into::map(v, f).collect() }
//#[macro_export] macro_rules! eval { ($($args:expr),*; |$($params:ident),*| $expr:expr) => { $crate::vec::eval($crate::zip!($($args,)*), |($($params),*)| $expr) }; }

pub use into::{IntoCopied as Copied, IntoChain as Chain, IntoZip as Zip, IntoMap as Map};

pub fn dot<A: std::ops::Mul<B>, B, R: std::iter::Sum<<A as std::ops::Mul<B>>::Output>>(iter: impl std::iter::IntoIterator<Item=(A,B)>) -> R {
	iter.into_iter().map(|(a,b)| a*b).sum::<R>()
}

pub trait Dot<B, R> { fn dot(self, other: B) -> R; }
impl<A: IntoExactSizeIterator+IntoIterator<Item: std::ops::Mul<<B as IntoIterator>::Item>>, B: IntoExactSizeIterator, R: std::iter::Sum<<<A as IntoIterator>::Item as std::ops::Mul<<B as IntoIterator>::Item>>::Output>> Dot<B, R> for A {
	fn dot(self, b: B) -> R { dot(self.zip(b)) }
}

pub trait DotN<B, R, const N: usize> { fn dot(self, other: B) -> R; }
impl<A: IntoConstSizeIterator<N>+IntoIterator<Item: std::ops::Mul<<B as IntoIterator>::Item>>, B: IntoConstSizeIterator<N>, R: std::iter::Sum<<<A as IntoIterator>::Item as std::ops::Mul<<B as IntoIterator>::Item>>::Output>, const N: usize> DotN<B, R, N> for A where Self:IntoConstSizeIterator<N>, B:IntoConstSizeIterator<N> {
	fn dot(self, b: B) -> R { dot(self.zip(b)) }
}

pub fn zip<A,B>(a: impl std::iter::IntoIterator<Item=A>, b: impl Fn(usize)->B) -> impl Iterator<Item=(A, B)> { a.into_iter().enumerate().map(move |(i,a)| (a,b(i))) }

pub fn list<T>(iter: impl std::iter::IntoIterator<Item=T>) -> Box<[T]> { iter.into_iter().collect() }
pub fn map<T, U>(iter: impl std::iter::IntoIterator<Item=T>, f: impl FnMut(T)->U) -> Box<[U]> { iter.into_iter().map(f).collect() }
