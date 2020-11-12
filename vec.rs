use super::{IntoIterator, ExactSizeIterator};

pub trait Vector<const N: usize> : Sized { 	type Item; }
impl<T:Vector<N>, const N: usize> Vector<N> for &'t T { type Item = &'t T::Item; }
impl<T, const N: usize> Vector<N> for [T; N] {	type Item = T; }
impl<V:Vector<N>, F:Fn<(V::Item,)>, const N: usize> Vector<N> for super::into::Map<V, F> { type Item = F::Output; }
impl<A:Vector<N>, B:Vector<N>, const N: usize> Vector<N> for super::into::Zip<A, B> {	type Item = (<A as Vector<N>>::Item, <B as Vector<N>>::Item); }

pub trait VectorCollect<const N: usize> : Vector<N> {
	fn collect(self) -> [Self::Item; N];
}
impl<T,V:Vector<N,Item=T>+IntoIterator<Item=T,IntoIter:ExactSizeIterator>, const N: usize> VectorCollect<N> for V {
	fn collect(self) -> [Self::Item; N] { super::FromExactSizeIterator::from_iter(self.into_iter()) }
}

pub fn eval<T, U, const N: usize>(v: impl Vector<N, Item=T>+IntoIterator<Item=T>+super::IntoExactSizeIterator, f: impl Fn(T)->U) -> [U; N] { VectorCollect::collect(super::into::IntoMap::map(v, f)) }
#[macro_export] macro_rules! eval { ($($args:expr),*; |$($params:ident),*| $expr:expr) => { $crate::vec::eval($crate::zip!($($args,)*), |($($params),*)| $expr) }; }

struct ConstRange<const N: usize>(std::ops::Range<usize>);
impl<const N: usize> ConstRange<N> { fn new() -> Self { Self(0..N) } }
impl<const N: usize> IntoIterator for ConstRange<N> { type IntoIter = std::ops::Range<usize>; type Item = <Self::IntoIter as Iterator>::Item; fn into_iter(self) -> Self::IntoIter { self.0 } }
impl<const N: usize> Vector<N> for ConstRange<N> { type Item = <Self as IntoIterator>::Item; }
#[track_caller] pub fn generate<T, F:Fn(usize)->T, const N:usize>(f : F) -> [T; N] { eval(ConstRange::new(), f) }

pub trait Sub<B=Self> { type Output; fn sub(self, b: B) -> Self::Output; }
impl<T: 't, const N: usize> Sub for &'t [T; N] where &'t T:std::ops::Sub {
	type Output = [<&'t T as std::ops::Sub>::Output; N];
	fn sub(self, b: &'t [T; N]) -> Self::Output { eval!(self, b; |a,b| a-b) }
}

pub trait Dot<T> { type Output; fn dot(self, other: T) -> Self::Output; }
impl<A: IntoIterator<Item:std::ops::Mul<B::Item>,IntoIter:'t>, B:IntoIterator<Item:>> Dot<B> for A where <A::Item as std::ops::Mul<B::Item>>::Output:std::iter::Sum {
	type Output = <A::Item as std::ops::Mul<B::Item>>::Output; fn dot(self, b: B) -> Self::Output { super::into::Sum::sum(super::map!(self, b; |a,b| a*b)) }
}

use std::convert::TryInto;
pub trait Suffix<T> { fn suffix<const S: usize>(&self) -> &[T; S]; }
impl<T, const N: usize> Suffix<T> for [T; N] { fn suffix<const S: usize>(&self) -> &[T; S] { (&self[N-S..]).try_into().unwrap() } }
