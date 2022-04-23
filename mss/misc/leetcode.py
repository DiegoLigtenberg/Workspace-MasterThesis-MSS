class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        longest_word = len(max(strs,key=len))
        lcp = 0
        str_copy = strs
        for i in (range(longest_word)):
            str_copy = [x[:longest_word-i] for x in str_copy]
            if all(ele == str_copy[0] for ele in str_copy): 
                lcp = longest_word-i; 
                break
        return strs[0][:lcp]
      

sol = Solution()

print(sol.longestCommonPrefix(["agja","agjatia","agjaea"]))
